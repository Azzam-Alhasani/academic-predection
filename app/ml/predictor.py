"""
Prediction interface for the Academic Performance Monitoring & Prediction System.
Manages model lifecycle: training on demand and running predictions.
"""

import os
from flask import current_app
from app.data.data_loader import load_students
from app.ml.preprocessing import preprocess, preprocess_single
from app.ml.trainer import train_all_models

# In-memory cache: avoids re-training on every request within a session
_model_cache = None


def _get_models():
    """Return cached models, training them if necessary."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    df = load_students()
    if df.empty or len(df) < 10:
        return None

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler, feature_names = preprocess(df)
    results = train_all_models(X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test)

    _model_cache = {
        "results": results,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    return _model_cache


def invalidate_cache():
    """Call this whenever student data changes so models retrain on next request."""
    global _model_cache
    _model_cache = None


def get_model_metrics() -> dict:
    """Return evaluation metrics for all models (no input required)."""
    cache = _get_models()
    if cache is None:
        return {}

    metrics = {}
    for key, result in cache["results"].items():
        entry = {
            "name": result["name"],
            "type": result["type"],
            "description": result["description"],
        }
        if result["type"] == "regression":
            entry["rmse"] = result["rmse"]
            entry["r2"] = result["r2"]
        else:
            entry["accuracy"] = result["accuracy"]
            entry["precision"] = result["precision"]
            entry["recall"] = result["recall"]
            entry["f1"] = result["f1"]
            entry["confusion_matrix"] = result["confusion_matrix"]
            entry["roc_auc"] = result.get("roc_auc")
        metrics[key] = entry
    return metrics


def predict_student(record: dict) -> dict:
    """
    Run all models on a single student record dict.

    Parameters
    ----------
    record : dict with keys matching student feature columns

    Returns
    -------
    dict with prediction results from each model
    """
    cache = _get_models()
    if cache is None:
        return {"error": "Not enough data to train models. Please add more students."}

    X = preprocess_single(record, cache["scaler"], cache["feature_names"])
    predictions = {}

    for key, result in cache["results"].items():
        model = result["model"]
        if result["type"] == "regression":
            score = float(model.predict(X)[0])
            predictions[key] = {
                "name": result["name"],
                "type": "regression",
                "predicted_score": round(score, 1),
                "grade": _score_to_grade(score),
            }
        else:
            label = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
            predictions[key] = {
                "name": result["name"],
                "type": "classification",
                "prediction": "Pass" if label == 1 else "Fail",
                "confidence": round(max(proba) * 100, 1) if proba else None,
            }

    return predictions


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
