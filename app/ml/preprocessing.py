"""
Preprocessing utilities for the ML pipeline.
Handles feature selection, encoding, scaling, and train/test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


FEATURE_COLS = [
    "attendance_pct",
    "assignments_submitted",
    "study_hours_per_week",
    "previous_gpa",
    "gender_encoded",
]

REGRESSION_TARGET = "avg_score"
CLASSIFICATION_TARGET = "pass_fail"


def preprocess(df: pd.DataFrame):
    """
    Prepare features and targets for training.

    Returns
    -------
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler, feature_names
    """
    df = df.copy()

    # Encode gender
    le = LabelEncoder()
    if "gender" in df.columns:
        df["gender_encoded"] = le.fit_transform(df["gender"].astype(str))
    else:
        df["gender_encoded"] = 0

    # Compute avg_score if not present
    subject_cols = ["math", "science", "english", "history", "cs"]
    present = [c for c in subject_cols if c in df.columns]
    if "avg_score" not in df.columns and present:
        df["avg_score"] = df[present].mean(axis=1)

    # Drop rows with missing targets
    df = df.dropna(subset=[REGRESSION_TARGET, CLASSIFICATION_TARGET])

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features].fillna(df[available_features].median())
    y_reg = df[REGRESSION_TARGET].values
    y_clf = df[CLASSIFICATION_TARGET].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_scaled, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler, available_features


def preprocess_single(record: dict, scaler: "StandardScaler", feature_names: list) -> np.ndarray:
    """Transform a single student record dict into a scaled feature vector."""
    le = LabelEncoder()
    le.classes_ = np.array(["Female", "Male"])
    gender_enc = 1 if str(record.get("gender", "Male")).strip().title() == "Male" else 0
    row = {}
    for feat in feature_names:
        if feat == "gender_encoded":
            row[feat] = gender_enc
        else:
            row[feat] = record.get(feat, 0)
    X = pd.DataFrame([row])[feature_names].fillna(0)
    return scaler.transform(X)
