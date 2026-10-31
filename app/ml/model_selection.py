"""
Model-selection study (run offline):   python -m app.ml.model_selection

Answers "which model should the system use, and why?" with evidence instead of a guess.

Method
  1. Stage A - every candidate with default settings, scored with 5-fold cross-validation
     repeated 3 times (15 scores each, the same splits for every model).
  2. Stage B - the strongest candidates (plus logistic regression as the simple reference)
     are tuned with NESTED cross-validation: the inner loop picks hyper-parameters, the outer
     loop scores them on students the tuning never saw, so tuning cannot flatter the result.
  3. Stage C - choice rule: highest mean score, then the *parsimony* (one-standard-error)
     rule - the simplest model within one standard error of the best is selected. Paired
     corrected resampled t-tests (Nadeau & Bengio, 2003) show which differences are real.
  4. The selected model is examined: out-of-fold predictions at several thresholds,
     calibration, permutation importance and a learning curve.

Results are written to app/ml/model_selection.json (shown on the Model Selection page) and
docs/09_model_selection.md. Takes roughly 15-30 minutes.
"""

import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, learning_curve, train_test_split,
)
from sklearn.pipeline import make_pipeline

from app.ml.preprocessing import (
    SCENARIOS, build_preprocessor, feature_columns, prepare_classification, prepare_regression,
)
from app.ml.registry import CLASSIFIERS, REGRESSORS, SEED, build_estimator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "model_selection.json")
OUT_MD = os.path.join(HERE, "..", "..", "docs", "09_model_selection.md")

QUICK = bool(os.environ.get("MS_QUICK"))           # MS_QUICK=1: small smoke test, nothing is saved
N_SPLITS, N_REPEATS_A, N_REPEATS_B = (3, 1, 1) if QUICK else (5, 3, 2)
INNER_FOLDS, INNER_ITER = (2, 2) if QUICK else (3, 5)
TEST_TRAIN_RATIO = 1 / (N_SPLITS - 1)          # test share / train share for Nadeau-Bengio
N_JOBS = 2


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ── metrics ──────────────────────────────────────────────────────────────────

def _probability(model, X):
    """Probability of the positive class; SVMs without probabilities use a sigmoid of the margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], True
    return 1 / (1 + np.exp(-model.decision_function(X))), False


def classification_metrics(y, p, calibrated=True, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p) if calibrated else float("nan"),
    }


def regression_metrics(y, pred) -> dict:
    return {
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
    }


def summarise(fold_metrics: list) -> dict:
    out = {}
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        if np.isnan(values).all():
            out[key] = {"mean": None, "sd": None}
        else:
            out[key] = {"mean": round(float(np.nanmean(values)), 4), "sd": round(float(np.nanstd(values, ddof=1)), 4)}
    return out


def corrected_t_test(a, b, ratio=TEST_TRAIN_RATIO) -> float:
    """Two-sided p-value, Nadeau & Bengio corrected resampled paired t-test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0
    t = d.mean() / math.sqrt((1 / k + ratio) * var)
    return float(2 * (1 - stats.t.cdf(abs(t), k - 1)))


# ── fold evaluation ──────────────────────────────────────────────────────────

def _fit_score_clf(pipe, X, y, train, test):
    model = clone(pipe).fit(X.iloc[train], y[train])
    p, calibrated = _probability(model, X.iloc[test])
    return classification_metrics(y[test], p, calibrated)


def _fit_score_reg(pipe, X, y, train, test):
    model = clone(pipe).fit(X.iloc[train], y[train])
    return regression_metrics(y[test], model.predict(X.iloc[test]))


def _nested_fold(pipe, grid, X, y, train, test, scenario, task):
    """One outer fold of nested CV: tune on `train` only, then score on `test`."""
    scoring = "roc_auc" if task == "clf" else "neg_root_mean_squared_error"
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED) if task == "clf" else \
        KFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipe, {f"{pipe.steps[-1][0]}__{k}": v for k, v in grid.items()}, n_iter=INNER_ITER,
        scoring=scoring, cv=inner, random_state=SEED, n_jobs=1, refit=True,
    )
    search.fit(X.iloc[train], y[train])
    best = search.best_estimator_
    if task == "clf":
        p, calibrated = _probability(best, X.iloc[test])
        return classification_metrics(y[test], p, calibrated)
    return regression_metrics(y[test], best.predict(X.iloc[test]))


def _make_pipe(scenario, registry, key, params=None):
    return make_pipeline(build_preprocessor(scenario), build_estimator(registry, key, params))


def _grid_size(grid):
    return int(np.prod([len(v) for v in grid.values()])) if grid else 1


# ── the study for one task / scenario ────────────────────────────────────────

def run_study(X, y, scenario, registry, task, score_key, higher_is_better, label):
    cv_a = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_A, random_state=SEED) if task == "clf" \
        else RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_A, random_state=SEED)
    splits_a = list(cv_a.split(X, y))
    fit = _fit_score_clf if task == "clf" else _fit_score_reg

    # Stage A: defaults, repeated CV
    log(f"[{label}] Stage A: default settings, {len(splits_a)} folds per model")
    stage_a = {}
    for key, spec in registry.items():
        pipe = _make_pipe(scenario, registry, key)
        folds = Parallel(n_jobs=N_JOBS)(delayed(fit)(pipe, X, y, tr, te) for tr, te in splits_a)
        stage_a[key] = {"folds": folds, "summary": summarise(folds)}
        log(f"   {spec[0]:32s} {score_key}={stage_a[key]['summary'][score_key]['mean']}")

    def mean_of(key, store):
        return store[key]["summary"][score_key]["mean"]

    real = [k for k in registry if registry[k][3] > 0]
    ranked = sorted(real, key=lambda k: mean_of(k, stage_a), reverse=higher_is_better)
    best_default = ranked[0]
    best_scores = [f[score_key] for f in stage_a[best_default]["folds"]]
    candidates = []
    for key in registry:
        scores = [f[score_key] for f in stage_a[key]["folds"]]
        candidates.append({
            "key": key, "label": registry[key][0], "description": registry[key][4], "complexity": registry[key][3],
            "metrics": stage_a[key]["summary"],
            "p_vs_best": None if key == best_default else round(corrected_t_test(best_scores, scores), 4),
            "is_best_default": key == best_default,
        })

    # Stage B: nested CV for the strongest tunable candidates
    top = [k for k in ranked[:4]]
    if "logistic" in registry and "logistic" not in top:
        top.append("logistic")
    if "linear" in registry and "linear" not in top:
        top.append("linear")
    cv_b = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_B, random_state=SEED + 1) if task == "clf" \
        else RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_B, random_state=SEED + 1)
    splits_b = list(cv_b.split(X, y))
    log(f"[{label}] Stage B: nested CV for {top} ({len(splits_b)} outer folds)")
    tuned = {}
    for key in top:
        pipe = _make_pipe(scenario, registry, key)
        grid = registry[key][2]
        if grid:
            folds = Parallel(n_jobs=N_JOBS)(
                delayed(_nested_fold)(pipe, grid, X, y, tr, te, scenario, task) for tr, te in splits_b)
        else:                                # nothing to tune: same as plain CV on these splits
            folds = Parallel(n_jobs=N_JOBS)(delayed(fit)(pipe, X, y, tr, te) for tr, te in splits_b)
        tuned[key] = {"folds": folds, "summary": summarise(folds)}
        log(f"   {registry[key][0]:32s} nested {score_key}={tuned[key]['summary'][score_key]['mean']}")

    # Stage C: choose with the parsimony rule
    pick = sorted(tuned, key=lambda k: mean_of(k, tuned), reverse=higher_is_better)
    best_tuned = pick[0]
    sd = tuned[best_tuned]["summary"][score_key]["sd"] or 0.0
    se = sd / math.sqrt(len(splits_b))
    best_mean = mean_of(best_tuned, tuned)
    if higher_is_better:
        within = [k for k in tuned if mean_of(k, tuned) >= best_mean - se]
    else:
        within = [k for k in tuned if mean_of(k, tuned) <= best_mean + se]
    selected = min(within, key=lambda k: (registry[k][3], -mean_of(k, tuned) if higher_is_better else mean_of(k, tuned)))
    best_t_scores = [f[score_key] for f in tuned[best_tuned]["folds"]]
    tuned_rows = []
    for key in pick:
        scores = [f[score_key] for f in tuned[key]["folds"]]
        tuned_rows.append({
            "key": key, "label": registry[key][0], "complexity": registry[key][3],
            "metrics": tuned[key]["summary"],
            "p_vs_best": None if key == best_tuned else round(corrected_t_test(best_t_scores, scores), 4),
            "within_one_se": key in within, "is_best": key == best_tuned,
        })
    log(f"[{label}] best tuned: {best_tuned}; within 1 SE: {within}; SELECTED: {selected}")

    # Final tuning of the selected model on all data (parameters stored for the live app)
    pipe = _make_pipe(scenario, registry, selected)
    grid = registry[selected][2]
    best_params = {}
    if grid:
        scoring = "roc_auc" if task == "clf" else "neg_root_mean_squared_error"
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED) if task == "clf" else KFold(5, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            pipe, {f"{pipe.steps[-1][0]}__{k}": v for k, v in grid.items()},
            n_iter=min(12, _grid_size(grid)), scoring=scoring, cv=inner, random_state=SEED, n_jobs=N_JOBS)
        search.fit(X, y)
        best_params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
        best_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}
    return {
        "candidates": candidates, "tuned": tuned_rows, "best_default": best_default,
        "best_tuned": best_tuned, "selected": selected, "selected_label": registry[selected][0],
        "selected_params": best_params, "se_used": round(se, 5),
        "rule": ("highest mean " + score_key + ", then the simplest model within one standard error of it"),
    }


def clf_examination(X, y, scenario, key, params):
    """Out-of-fold behaviour of the selected classifier."""
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
    pipe = _make_pipe(scenario, CLASSIFIERS, key, params)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    method = "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"
    raw = cross_val_predict(pipe, X, y, cv=cv, method=method, n_jobs=N_JOBS)
    p = raw[:, 1] if method == "predict_proba" else 1 / (1 + np.exp(-raw))

    thresholds = []
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        m = classification_metrics(y, p, True, t)
        thresholds.append({"threshold": t, **{k: round(m[k], 3) for k in ("accuracy", "precision", "recall", "f1")}})
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    calibration = [{"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "predicted": round(float(p[idx == i].mean()), 3),
                    "observed": round(float(y[idx == i].mean()), 3), "n": int((idx == i).sum())}
                   for i in range(10) if (idx == i).sum() >= 10]
    pred = (p >= 0.5).astype(int)
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tp = int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    model = clone(pipe).fit(Xtr, ytr)
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=N_JOBS)
    order = np.argsort(-imp.importances_mean)[:10]
    importance = [{"feature": X.columns[i], "drop_in_auc": round(float(imp.importances_mean[i]), 4),
                   "sd": round(float(imp.importances_std[i]), 4)} for i in order]

    sizes, train_s, test_s = learning_curve(
        pipe, X, y, cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED), scoring="roc_auc",
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=N_JOBS, random_state=SEED)
    return {
        "thresholds": thresholds, "calibration": calibration, "brier": round(float(brier_score_loss(y, p)), 4),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "importance": importance,
        "learning_curve": {"sizes": [int(s) for s in sizes], "train": [round(float(v), 4) for v in train_s.mean(1)],
                           "validation": [round(float(v), 4) for v in test_s.mean(1)]},
    }


def reg_examination(X, y, key, params):
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()}
    pipe = _make_pipe("after_sem1", REGRESSORS, key, params)
    pred = cross_val_predict(pipe, X, y, cv=KFold(N_SPLITS, shuffle=True, random_state=SEED), n_jobs=N_JOBS)
    resid = y - pred
    return {"within_5_marks": round(float((np.abs(resid) <= 5).mean()), 3),
            "within_10_marks": round(float((np.abs(resid) <= 10).mean()), 3),
            "mean_error": round(float(resid.mean()), 3)}


# ── report ───────────────────────────────────────────────────────────────────

def write_markdown(result):
    def pm(m):
        return "n/a" if m["mean"] is None else f"{m['mean']:.3f} ± {m['sd']:.3f}"

    lines = ["# 09 - Model selection analysis", "",
             "_Generated by `python -m app.ml.model_selection`; the same numbers are on the Model Selection page._", "",
             f"Dataset: {result['meta']['n_students']} real university students "
             f"(UCI *Predict Students' Dropout and Academic Success*), dropout rate {result['meta']['dropout_rate']:.1%}.", "",
             "## Method", "",
             f"1. **Stage A**: {len(CLASSIFIERS)} classifiers, default settings, {N_SPLITS}-fold cross-validation repeated "
             f"{N_REPEATS_A} times ({N_SPLITS * N_REPEATS_A} scores per model, identical splits).",
             f"2. **Stage B**: the strongest candidates plus logistic regression are tuned with *nested* cross-validation "
             f"({N_SPLITS}-fold outer x {N_REPEATS_B} repeats; inner {INNER_FOLDS}-fold random search, {INNER_ITER} settings).",
             "3. **Stage C**: choose the highest mean ROC-AUC, then apply the one-standard-error parsimony rule "
             "(simplest model within one standard error of the best). `p` values are corrected resampled paired t-tests "
             "(Nadeau & Bengio, 2003) against the best model; p < 0.05 means the difference is unlikely to be chance.", ""]
    for scen, block in result["classification"].items():
        lines += [f"## Dropout classification - {SCENARIOS[scen]['label']}", "",
                  "### Stage A (default settings)", "",
                  "| Model | Accuracy | Balanced acc. | Recall | F1 | ROC-AUC | p vs best |", "|---|---|---|---|---|---|---|"]
        for c in sorted(block["candidates"], key=lambda c: -(c["metrics"]["roc_auc"]["mean"] or 0)):
            m = c["metrics"]
            lines.append(f"| {c['label']} | {pm(m['accuracy'])} | {pm(m['balanced_accuracy'])} | {pm(m['recall'])} | "
                         f"{pm(m['f1'])} | {pm(m['roc_auc'])} | {'best' if c['is_best_default'] else c['p_vs_best']} |")
        lines += ["", "### Stage B (tuned, nested CV)", "",
                  "| Model | Accuracy | Recall | F1 | ROC-AUC | p vs best | Within 1 SE |", "|---|---|---|---|---|---|---|"]
        for c in block["tuned"]:
            m = c["metrics"]
            lines.append(f"| {c['label']} | {pm(m['accuracy'])} | {pm(m['recall'])} | {pm(m['f1'])} | {pm(m['roc_auc'])} | "
                         f"{'best' if c['is_best'] else c['p_vs_best']} | {'yes' if c['within_one_se'] else 'no'} |")
        lines += ["", f"**Selected: {block['selected_label']}** - parameters {block['selected_params'] or 'defaults'}.", ""]
    reg = result["regression"]
    lines += ["## Semester-2 mark regression (after semester 1)", "", "| Model | RMSE | MAE | R² |", "|---|---|---|---|"]
    for c in reg["tuned"]:
        m = c["metrics"]
        lines.append(f"| {c['label']} | {pm(m['rmse'])} | {pm(m['mae'])} | {pm(m['r2'])} |")
    lines += ["", f"**Selected: {reg['selected_label']}.**", ""]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    from flask import Flask
    import config
    from app.data.data_loader import load_students

    app = Flask(__name__)
    app.config.from_object(config.DevelopmentConfig)
    with app.app_context():
        df = load_students()
    if QUICK:
        df = df.sample(900, random_state=SEED).reset_index(drop=True)
    start = time.time()
    result = {"meta": {
        "n_students": int(len(df)), "dropout_rate": float(df["dropout"].mean()),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "scikit_learn": sklearn.__version__, "python": platform.python_version(), "seed": SEED,
        "cv": {"splits": N_SPLITS, "repeats_stage_a": N_REPEATS_A, "repeats_stage_b": N_REPEATS_B,
               "inner_folds": INNER_FOLDS, "inner_candidates": INNER_ITER},
    }, "classification": {}, "regression": {}}

    for scenario in SCENARIOS:
        X, y = prepare_classification(df, scenario)
        block = run_study(X, y, scenario, CLASSIFIERS, "clf", "roc_auc", True, f"classification/{scenario}")
        block["n"] = int(len(y))
        block["positive_rate"] = round(float(y.mean()), 4)
        block["examination"] = clf_examination(X, y, scenario, block["selected"], block["selected_params"])
        result["classification"][scenario] = block
        if not QUICK:
            with open(OUT_JSON, "w") as f:
                json.dump(result, f, indent=1)

    X, y = prepare_regression(df)
    block = run_study(X, y, "after_sem1", REGRESSORS, "reg", "rmse", False, "regression/s2_mark")
    block["n"] = int(len(y))
    block["target_sd"] = round(float(y.std()), 3)
    block["examination"] = reg_examination(X, y, block["selected"], block["selected_params"])
    result["regression"] = block
    result["meta"]["minutes"] = round((time.time() - start) / 60, 1)
    if QUICK:
        log("quick test finished OK (nothing saved)")
        return
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    write_markdown(result)
    log("done ->", OUT_JSON)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
    main()
