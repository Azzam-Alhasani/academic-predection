"""
Feature definitions and preprocessing for the ML pipeline.

Only the factors that matter for results are used: the term marks, the number of absences and
the course. Two prediction moments (scenarios) are supported, because a warning is only useful
if it comes early:
  * after_term1 - term 1 mark + absences + course
  * after_term2 - term 1 mark, term 2 mark, the change between them + absences + course

Targets (known only for records that have a final mark)
  * classification: fail (1) = final mark below the 50% pass mark, pass (0)
  * regression:     the final mark (0-100)
The final mark is never used as an input. `person_id` is only used to keep a student's two
courses together in train/test splits (never a model input).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["course"]

SCENARIOS = {
    "after_term1": {"label": "After term 1", "numeric": ["t1_mark", "absences"], "categorical": CATEGORICAL},
    "after_term2": {"label": "After term 2", "numeric": ["t1_mark", "t2_mark", "trend", "absences"],
                    "categorical": CATEGORICAL},
}

CLASSIFICATION_TARGET = "fail"
REGRESSION_TARGET = "final_mark"
SEED = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the change between term 1 and term 2 (a model input)."""
    df = df.copy()
    for col in ("t1_mark", "t2_mark"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["trend"] = df["t2_mark"] - df["t1_mark"]
    return df


def feature_columns(scenario: str) -> list:
    spec = SCENARIOS[scenario]
    return spec["numeric"] + spec["categorical"]


def build_preprocessor(scenario: str) -> ColumnTransformer:
    """Impute missing values, scale numbers, one-hot encode the course (dense output)."""
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


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    df = add_features(df)
    return df[pd.to_numeric(df["final_mark"], errors="coerce").notna()].reset_index(drop=True)


def prepare_classification(df: pd.DataFrame, scenario: str):
    """(X, y, groups): y = 1 for a failed course; groups link one student's courses."""
    df = _labelled(df)
    y = (pd.to_numeric(df["final_mark"]) < 50).astype(int).to_numpy()
    return df[feature_columns(scenario)], y, df["person_id"].to_numpy()


def prepare_regression(df: pd.DataFrame, scenario: str):
    """(X, y, groups): y = the final mark."""
    df = _labelled(df)
    return df[feature_columns(scenario)], pd.to_numeric(df["final_mark"]).to_numpy(float), df["person_id"].to_numpy()


def record_to_frame(record: dict, scenario: str) -> pd.DataFrame:
    """One-row feature frame from a form/record dict; missing fields become NaN (imputed)."""
    frame = add_features(pd.DataFrame([{k: (np.nan if v in (None, "") else v) for k, v in record.items()}]))
    for col in feature_columns(scenario):
        if col not in frame.columns:
            frame[col] = np.nan
    for col in SCENARIOS[scenario]["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[feature_columns(scenario)]
