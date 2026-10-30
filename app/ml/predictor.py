"""
Prediction interface: trains the models on demand (cached) and runs predictions.
"""

import threading

from app.data.data_loader import load_students, _score_to_grade
from app.ml.preprocessing import prepare, record_to_frame
from app.ml.trainer import train_all_models

MIN_STUDENTS = 30

_model_cache = None
_lock = threading.Lock()


def _get_models():
    """Return cached models, training them on first use (thread-safe)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _lock:
        if _model_cache is not None:
            return _model_cache
        df = load_students()
        if df.empty or len(df) < MIN_STUDENTS:
            return None
        X, y_reg, y_clf, groups = prepare(df)
        if len(set(y_clf)) < 2:
            return None  # need both passing and failing students to learn from
        _model_cache = {
            "results": train_all_models(X, y_reg, y_clf, groups),
            "n_students": int(len(X)),
        }
        return _model_cache


def invalidate_cache():
    """Call whenever student data changes so models retrain on the next request."""
    global _model_cache
    with _lock:
        _model_cache = None


def get_model_metrics() -> dict:
    """Return cross-validated evaluation metrics for all models."""
    cache = _get_models()
    if cache is None:
        return {}

    metrics = {}
    for key, result in cache["results"].items():
        entry = {"name": result["name"], "type": result["type"], "description": result["description"]}
        if result["type"] == "regression":
            entry.update(rmse=result["rmse"], r2=result["r2"])
        else:
            entry.update(
                accuracy=result["accuracy"], precision=result["precision"],
                recall=result["recall"], f1=result["f1"],
                confusion_matrix=result["confusion_matrix"], roc_auc=result.get("roc_auc"),
                baseline=result["baseline"],
            )
        metrics[key] = entry
    return metrics


def get_training_size() -> int:
    cache = _get_models()
    return cache["n_students"] if cache else 0


def predict_student(record: dict) -> dict:
    """Run all models on a single student record (dict of form/record fields)."""
    cache = _get_models()
    if cache is None:
        return {"error": f"Not enough data to train models (need at least {MIN_STUDENTS} students)."}

    X = record_to_frame(record)
    predictions = {}
    for key, result in cache["results"].items():
        model = result["model"]
        if result["type"] == "regression":
            score = float(min(20.0, max(0.0, model.predict(X)[0])))
            predictions[key] = {
                "name": result["name"],
                "type": "regression",
                "predicted_score": round(score, 1),
                "grade": _score_to_grade(score / 20 * 100),
            }
        else:
            p_pass = float(model.predict_proba(X)[0][1])
            predictions[key] = {
                "name": result["name"],
                "type": "classification",
                "prediction": "Pass" if p_pass >= 0.5 else "Fail",
                "confidence": round(max(p_pass, 1 - p_pass) * 100, 1),
                "p_pass": round(p_pass * 100, 1),
            }
    return predictions
