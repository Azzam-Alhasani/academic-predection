import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def train_all_models(X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test):
    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)
    y_pred_lr = lr.predict(X_test)
    results["linear_regression"] = {
        "model": lr,
        "type": "regression",
        "name": "Linear Regression",
        "rmse": round(float(np.sqrt(mean_squared_error(y_reg_test, y_pred_lr))), 3),
        "r2": round(float(r2_score(y_reg_test, y_pred_lr)), 3),
        "description": "Predicts the final average score (continuous value) based on student features.",
    }

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_clf_train)
    results["logistic_regression"] = {
        "model": log_reg,
        "type": "classification",
        "name": "Logistic Regression",
        **_clf_metrics(y_clf_test, log_reg.predict(X_test), X_test, log_reg),
        "description": "Predicts whether a student will pass or fail (binary classification).",
    }

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_clf_train)
    results["random_forest"] = {
        "model": rf_clf,
        "type": "classification",
        "name": "Random Forest",
        **_clf_metrics(y_clf_test, rf_clf.predict(X_test), X_test, rf_clf),
        "description": "Ensemble of decision trees for robust pass/fail classification.",
    }

    dt_clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt_clf.fit(X_train, y_clf_train)
    results["decision_tree"] = {
        "model": dt_clf,
        "type": "classification",
        "name": "Decision Tree",
        **_clf_metrics(y_clf_test, dt_clf.predict(X_test), X_test, dt_clf),
        "description": "Rule-based tree model for interpretable pass/fail prediction.",
    }

    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_clf_train)
    results["naive_bayes"] = {
        "model": nb_clf,
        "type": "classification",
        "name": "Naive Bayes",
        **_clf_metrics(y_clf_test, nb_clf.predict(X_test), X_test, nb_clf),
        "description": "Probabilistic classifier using Bayes' theorem with Gaussian features.",
    }

    return results


def _clf_metrics(y_true, y_pred, X_test=None, model=None) -> dict:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 3)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics