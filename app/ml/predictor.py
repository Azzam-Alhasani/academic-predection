"""
Prediction interface used by the web app.

The offline study (app/ml/model_selection.py) chooses the model and its settings and stores them in
app/ml/model_selection.json. Here the chosen models are fitted on the current data (a few seconds)
and cached; quality figures always come from the study, not from the live fit.

For every course attempt the models give a predicted final mark and the probability of failing
(grade below C), at two moments: "planning" (no attendance yet) and "during" (with attendance %).
"""

import json
import os
import threading

import numpy as np
from sklearn.pipeline import make_pipeline

from app.data.data_loader import load_tables
from app.ml.preprocessing import (
    SCENARIOS, build_course_frame, build_preprocessor, prepare_classification, prepare_regression, records_to_frame,
)
from app.ml.registry import CLASSIFIERS, REGRESSORS, build_estimator

MIN_ROWS = 300
SELECTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_selection.json")

_cache = {"tables": None, "models": None, "frame": None}
_selection_cache = None
_lock = threading.Lock()


def load_selection() -> dict:
    """The stored model-selection study (empty dict if it has not been run)."""
    global _selection_cache
    if _selection_cache is None:
        try:
            with open(SELECTION_FILE, encoding="utf-8") as f:
                _selection_cache = json.load(f)
        except (OSError, ValueError):
            _selection_cache = {}
    return _selection_cache


def _params(block) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (block or {}).get("selected_params", {}).items()}


def _pipeline(scenario, registry, key, params=None):
    return make_pipeline(build_preprocessor(scenario), build_estimator(registry, key, params))


def get_frame():
    """The course-attempt frame behind the current tables (cached together with the models)."""
    _get_models()
    return _cache["frame"]


def _get_models():
    """Fit (once per data version) and return the live models, or None when there is too little data."""
    tables = load_tables()
    if _cache["tables"] is tables and _cache["models"] is not None:
        return _cache["models"]
    with _lock:
        tables = load_tables()
        if _cache["tables"] is tables and _cache["models"] is not None:
            return _cache["models"]
        if tables.empty:
            return None
        frame = build_course_frame(tables)
        if int((frame["attempted"] > 0).sum()) < MIN_ROWS:
            return None
        study = load_selection()
        models = {"classifier": {}, "regressor": {}}
        for scenario in SCENARIOS:
            X, y, _ = prepare_classification(frame, scenario)
            block = study.get("classification", {}).get(scenario)
            key = block["selected"] if block else "logistic"
            models["classifier"][scenario] = {"key": key, "label": CLASSIFIERS[key][0],
                                              "pipeline": _pipeline(scenario, CLASSIFIERS, key, _params(block)).fit(X, y)}
            Xr, yr, _ = prepare_regression(frame, scenario)
            rblock = study.get("regression", {}).get(scenario)
            rkey = rblock["selected"] if rblock else "ridge"
            models["regressor"][scenario] = {"key": rkey, "label": REGRESSORS[rkey][0],
                                             "pipeline": _pipeline(scenario, REGRESSORS, rkey, _params(rblock)).fit(Xr, yr)}
        _cache.update(tables=tables, models=models, frame=frame)
        return models


def invalidate_cache():
    """Call whenever data changes so the live models refit on the next request."""
    _cache.update(tables=None, models=None, frame=None)


def get_training_size() -> int:
    models = _get_models()
    return int((_cache["frame"]["attempted"] > 0).sum()) if models else 0


def regression_sd(scenario: str = "during") -> float:
    """Typical error (RMSE, in marks) of the selected regression model, from the study."""
    block = load_selection().get("regression", {}).get(scenario)
    if block:
        chosen = next((c for c in block["tuned"] if c["key"] == block["selected"]), None)
        if chosen and chosen["metrics"]["rmse"]["mean"]:
            return float(chosen["metrics"]["rmse"]["mean"])
    return 8.0


def model_labels(scenario: str = "during") -> dict:
    models = _get_models()
    if not models:
        return {}
    return {"classifier": models["classifier"][scenario]["label"], "regressor": models["regressor"][scenario]["label"]}


def predict_courses(records: list, scenario: str = "during"):
    """List of {mark, p_fail} (one per record), or None when there is not enough data."""
    models = _get_models()
    if models is None or not records:
        return None if models is None else []
    X = records_to_frame(records, scenario)
    marks = models["regressor"][scenario]["pipeline"].predict(X)
    pipe = models["classifier"][scenario]["pipeline"]
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X)[:, 1]
    else:
        p = 1 / (1 + np.exp(-pipe.decision_function(X)))
    return [{"mark": float(np.clip(m, 0, 100)), "p_fail": float(q)} for m, q in zip(marks, p)]
