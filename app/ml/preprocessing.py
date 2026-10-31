"""
Feature definitions and preprocessing for the ML pipeline.

Two prediction moments (scenarios) are supported, because a withdrawal warning is only
useful if it comes early:
  * after_sem1 - background + first-semester results
  * after_sem2 - background + first- and second-semester results

Targets
  * classification: dropout (1) vs graduate / still enrolled (0)
  * regression:     second-semester mark (0-100), predicted from the after_sem1 features
The final outcome is never used as an input.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BACKGROUND_NUM = ["age", "admission_mark", "prev_mark"]
BACKGROUND_CAT = ["gender", "programme", "attendance", "scholarship", "tuition_paid", "debtor"]
S1_NUM = ["s1_enrolled", "s1_approved", "s1_mark", "s1_completion"]
S2_NUM = ["s2_enrolled", "s2_approved", "s2_mark", "s2_completion"]

SCENARIOS = {
    "after_sem1": {
        "label": "After semester 1",
        "numeric": BACKGROUND_NUM + S1_NUM,
        "categorical": BACKGROUND_CAT,
    },
    "after_sem2": {
        "label": "After semester 2",
        "numeric": BACKGROUND_NUM + S1_NUM + S2_NUM,
        "categorical": BACKGROUND_CAT,
    },
}

CLASSIFICATION_TARGET = "dropout"
REGRESSION_TARGET = "s2_mark"
SEED = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-semester completion rates used as model inputs."""
    df = df.copy()
    for sem in ("s1", "s2"):
        enrolled = pd.to_numeric(df[f"{sem}_enrolled"], errors="coerce")
        approved = pd.to_numeric(df[f"{sem}_approved"], errors="coerce")
        df[f"{sem}_completion"] = np.where(enrolled > 0, approved / enrolled.where(enrolled > 0, 1), 0.0)
    return df


def feature_columns(scenario: str) -> list:
    spec = SCENARIOS[scenario]
    return spec["numeric"] + spec["categorical"]


def build_preprocessor(scenario: str) -> ColumnTransformer:
    """Impute missing values, scale numbers, one-hot encode categories (dense output)."""
    spec = SCENARIOS[scenario]
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer([
        ("num", numeric, spec["numeric"]),
        ("cat", categorical, spec["categorical"]),
    ])


def prepare_classification(df: pd.DataFrame, scenario: str):
    """(X, y) with y = 1 for students who withdrew."""
    df = add_features(df)
    return df[feature_columns(scenario)], df[CLASSIFICATION_TARGET].astype(int).to_numpy()


def prepare_regression(df: pd.DataFrame):
    """(X, y): rows where the student passed courses in both semesters; y = semester-2 mark."""
    df = add_features(df)
    df = df[(df["s1_approved"] > 0) & (df["s2_approved"] > 0)]
    return df[feature_columns("after_sem1")], df[REGRESSION_TARGET].astype(float).to_numpy()


def record_to_frame(record: dict, scenario: str) -> pd.DataFrame:
    """One-row feature frame from a form/record dict; missing fields become NaN (imputed)."""
    frame = add_features(pd.DataFrame([{k: (np.nan if v in (None, "") else v) for k, v in record.items()}]))
    for col in feature_columns(scenario):
        if col not in frame.columns:
            frame[col] = np.nan
    for col in SCENARIOS[scenario]["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[feature_columns(scenario)]
