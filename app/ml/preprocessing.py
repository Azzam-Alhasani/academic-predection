"""
Feature definition and preprocessing for the ML pipeline.

Predicting the FINAL result (G3) from the two earlier period grades (G1, G2) plus a
few habit/background fields. G3 itself is never used as an input.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PASS_MARK = 10

NUMERIC_FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"]
CATEGORICAL_FEATURES = ["subject", "gender", "higher", "internet"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REGRESSION_TARGET = "G3"
CLASSIFICATION_TARGET = "pass_fail"

# The same student can appear in both the Math and Portuguese files. These background
# attributes identify a student (per the dataset authors), so records sharing them are
# kept in the same cross-validation fold; otherwise the test would partly "see" the
# same student during training and look better than it really is.
GROUP_KEYS = [
    "school", "gender", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values, scale numbers, one-hot encode categories (dense output)."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group id per row; rows with missing key fields (added by hand) get their own."""
    keys = [k for k in GROUP_KEYS if k in df.columns]
    if not keys:
        return np.arange(len(df))
    text = df[keys].astype(str).agg("|".join, axis=1)
    incomplete = df[keys].isna().any(axis=1)
    if "student_id" in df.columns:
        text = text.where(~incomplete, "id-" + df["student_id"].astype(str))
    return text.astype("category").cat.codes.to_numpy()


def prepare(df: pd.DataFrame):
    """Return X (feature frame), regression target, classification target and groups."""
    df = df.dropna(subset=[REGRESSION_TARGET]).reset_index(drop=True)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    X = df[FEATURES].copy()
    y_reg = df[REGRESSION_TARGET].astype(float).to_numpy()
    y_clf = (df[REGRESSION_TARGET] >= PASS_MARK).astype(int).to_numpy()
    return X, y_reg, y_clf, make_groups(df)


def record_to_frame(record: dict) -> pd.DataFrame:
    """One-row feature frame from a form/record dict; missing fields become NaN (imputed)."""
    row = {}
    for col in FEATURES:
        value = record.get(col)
        row[col] = np.nan if value in (None, "") else value
    frame = pd.DataFrame([row])
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame
