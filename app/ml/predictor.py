"""
Prediction interface used by the web app.

The offline study (app/ml/model_selection.py) chooses the model and its settings and stores
them in app/ml/model_selection.json. Here the chosen models are fitted on the current data
(a second or two) and cached. Quality figures always come from the study, which used
cross-validation on the original dataset; they are not recomputed on every request.
"""

import json
import os
import threading

import numpy as np
from sklearn.pipeline import make_pipeline

from app.data.data_loader import load_students
from app.ml.preprocessing import (
    SCENARIOS, build_preprocessor, prepare_classification, prepare_regression, record_to_frame,
)
from app.ml.registry import CLASSIFIERS, REGRESSORS, build_estimator

MIN_STUDENTS = 100
SELECTION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_selection.json")

FEATURE_LABELS = {
    "age": "Age at enrolment", "admission_mark": "Admission mark", "prev_mark": "Previous qualification mark",
    "s1_enrolled": "Courses taken in semester 1", "s1_approved": "Courses passed in semester 1",
    "s1_mark": "Semester 1 mark", "s1_completion": "Semester 1 completion rate",
    "s2_enrolled": "Courses taken in semester 2", "s2_approved": "Courses passed in semester 2",
    "s2_mark": "Semester 2 mark", "s2_completion": "Semester 2 completion rate",
    "gender": "Gender", "programme": "Programme", "attendance": "Daytime / evening",
    "scholarship": "Scholarship", "tuition_paid": "Tuition fees", "debtor": "Debt",
}

_model_cache = None
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


def _params(block: dict) -> dict:
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in (block or {}).get("selected_params", {}).items()}


def _pipeline(scenario, registry, key, params=None):
    return make_pipeline(build_preprocessor(scenario), build_estimator(registry, key, params))


def _get_models():
    """Fit (once) and return the live models, or None when there is too little data."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _lock:
        if _model_cache is not None:
            return _model_cache
        df = load_students()
        if df.empty or len(df) < MIN_STUDENTS or df["dropout"].nunique() < 2:
            return None
        study = load_selection()
        models = {"classifier": {}, "explainer": {}, "regressor": None, "n": int(len(df))}
        for scenario in SCENARIOS:
            X, y = prepare_classification(df, scenario)
            block = study.get("classification", {}).get(scenario)
            key = block["selected"] if block else "logistic"
            params = _params(block) if block else {}
            models["classifier"][scenario] = {
                "key": key, "label": CLASSIFIERS[key][0],
                "pipeline": _pipeline(scenario, CLASSIFIERS, key, params).fit(X, y),
            }
            models["explainer"][scenario] = _pipeline(scenario, CLASSIFIERS, "logistic").fit(X, y)
        reg_block = study.get("regression")
        key = reg_block["selected"] if reg_block else "ridge"
        Xr, yr = prepare_regression(df)
        if len(yr) >= MIN_STUDENTS:
            models["regressor"] = {
                "key": key, "label": REGRESSORS[key][0],
                "pipeline": _pipeline("after_sem1", REGRESSORS, key, _params(reg_block) if reg_block else {}).fit(Xr, yr),
            }
        _model_cache = models
        return _model_cache


def invalidate_cache():
    """Call whenever student data changes so the live models refit on the next request."""
    global _model_cache
    with _lock:
        _model_cache = None


def get_training_size() -> int:
    models = _get_models()
    return models["n"] if models else 0


def risk_level(p_dropout: float) -> str:
    return "High" if p_dropout >= 50 else "Medium" if p_dropout >= 20 else "Low"


def _factors(explainer, X, scenario, top=3) -> dict:
    """Which inputs push this student's risk up or down (logistic-regression contributions)."""
    pre, clf = explainer.steps[0][1], explainer.steps[-1][1]
    row = pre.transform(X)[0]
    names = pre.get_feature_names_out()
    cats = SCENARIOS[scenario]["categorical"]
    totals = {}
    for name, value, coef in zip(names, row, clf.coef_[0]):
        bare = name.split("__", 1)[1]
        feature = next((c for c in cats if bare.startswith(c + "_")), bare)
        totals[feature] = totals.get(feature, 0.0) + float(value) * float(coef)
    ranked = sorted(totals.items(), key=lambda kv: kv[1])
    up = [(FEATURE_LABELS.get(f, f), round(v, 2)) for f, v in reversed(ranked) if v > 0.05][:top]
    down = [(FEATURE_LABELS.get(f, f), round(-v, 2)) for f, v in ranked if v < -0.05][:top]
    return {"raises": up, "lowers": down}


def predict_student(record: dict) -> dict:
    """Withdrawal risk (and projected semester-2 mark when possible) for one student record."""
    models = _get_models()
    if models is None:
        return {"error": f"Not enough data to train the models (need at least {MIN_STUDENTS} students)."}

    has_s2 = float(record.get("s2_enrolled") or 0) > 0
    scenario = "after_sem2" if has_s2 else "after_sem1"
    X = record_to_frame(record, scenario)
    pipe = models["classifier"][scenario]["pipeline"]
    if hasattr(pipe, "predict_proba"):
        p = float(pipe.predict_proba(X)[0][1])
    else:
        p = float(1 / (1 + np.exp(-pipe.decision_function(X)[0])))
    result = {
        "scenario": scenario, "scenario_label": SCENARIOS[scenario]["label"],
        "model": models["classifier"][scenario]["label"],
        "p_dropout": round(p * 100, 1), "level": risk_level(p * 100),
        "factors": _factors(models["explainer"][scenario], X, scenario),
        "projected_s2_mark": None,
    }
    reg = models["regressor"]
    if reg and not has_s2 and float(record.get("s1_approved") or 0) > 0:
        mark = float(reg["pipeline"].predict(record_to_frame(record, "after_sem1"))[0])
        result["projected_s2_mark"] = round(min(100.0, max(0.0, mark)), 1)
        result["projection_model"] = reg["label"]
    return result
