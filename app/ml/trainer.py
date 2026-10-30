"""
Train and evaluate all models.

Reported metrics come from 5-fold *grouped* cross-validation: every student is predicted
by a model that never saw that student (or the same student's record in the other
subject). The final models used for predictions are then fitted on all the data.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from app.ml.preprocessing import build_preprocessor

N_FOLDS = 5
SEED = 42


def _classifiers():
    return [
        ("logistic_regression", "Logistic Regression",
         "Predicts whether a student will pass or fail (binary classification).",
         LogisticRegression(max_iter=3000, random_state=SEED)),
        ("random_forest", "Random Forest",
         "Ensemble of decision trees for robust pass/fail classification.",
         RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1)),
        ("decision_tree", "Decision Tree",
         "Rule-based tree model for interpretable pass/fail prediction.",
         DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("naive_bayes", "Naive Bayes",
         "Probabilistic classifier using Bayes' theorem with Gaussian features.",
         GaussianNB()),
    ]


def train_all_models(X, y_reg, y_clf, groups):
    """Return {key: result dict}; each has the fitted pipeline under "model"."""
    results = {}
    baseline = float(max(np.mean(y_clf), 1 - np.mean(y_clf)))

    # ── Regression: predict the final grade G3 (0-20) ─────────────────────────
    reg = make_pipeline(build_preprocessor(), LinearRegression())
    y_pred = cross_val_predict(reg, X, y_reg, cv=GroupKFold(n_splits=N_FOLDS), groups=groups)
    y_pred = np.clip(y_pred, 0, 20)
    reg.fit(X, y_reg)
    results["linear_regression"] = {
        "model": reg,
        "type": "regression",
        "name": "Linear Regression",
        "rmse": round(float(np.sqrt(mean_squared_error(y_reg, y_pred))), 3),
        "r2": round(float(r2_score(y_reg, y_pred)), 3),
        "description": "Predicts the final grade G3 (0-20) from earlier grades and study habits.",
    }

    # ── Classification: predict pass / fail ───────────────────────────────────
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for key, name, description, estimator in _classifiers():
        pipe = make_pipeline(build_preprocessor(), estimator)
        proba = cross_val_predict(pipe, X, y_clf, cv=cv, groups=groups, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        pipe.fit(X, y_clf)
        results[key] = {
            "model": pipe,
            "type": "classification",
            "name": name,
            "description": description,
            "baseline": round(baseline, 3),
            **_clf_metrics(y_clf, pred, proba),
        }
    return results


def _clf_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        auc = round(float(roc_auc_score(y_true, y_prob)), 3)
    except ValueError:
        auc = None
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "roc_auc": auc,
    }
