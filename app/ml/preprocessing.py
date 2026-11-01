"""
Feature definitions and preprocessing for the ML pipeline.

One training row = one student's attempt at one course that finished with a mark (not W).
Two prediction moments (scenarios):
  * planning - known at registration: the student's CGPA and credits so far, the course (its code,
               credit hours, level), the semester's total credit load, the attempt number, the major
  * during   - the same plus the student's attendance % in that course

Targets: fail (1 = result F, i.e. grade below C) and the final mark (0-100).
Gender is stored for reporting but is deliberately NOT a model input.
`student_id` is only used to keep one student's rows together in train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = ["prior_cgpa", "prior_credits", "credit_hours", "load_credits", "level", "attempt"]
CATEGORICAL = ["course_code", "major"]

SCENARIOS = {
    "planning": {"label": "At registration (before classes)", "numeric": BASE_NUMERIC, "categorical": CATEGORICAL},
    "during": {"label": "During the semester (with attendance %)", "numeric": BASE_NUMERIC + ["attendance"],
               "categorical": CATEGORICAL},
}
SEED = 42


def build_course_frame(tables) -> pd.DataFrame:
    """One row per enrollment with the model inputs (history before that semester) and the targets."""
    enr = tables.enrollments.merge(tables.students[["student_id", "major"]], on="student_id", how="left")
    agg = (enr.groupby(["student_id", "sem_order"], as_index=False)[["quality", "attempted"]].sum()
              .sort_values(["student_id", "sem_order"]))
    agg["cq"] = agg.groupby("student_id")["quality"].cumsum() - agg["quality"]
    agg["ca"] = agg.groupby("student_id")["attempted"].cumsum() - agg["attempted"]
    enr = enr.merge(agg[["student_id", "sem_order", "cq", "ca"]], on=["student_id", "sem_order"], how="left")
    enr["prior_credits"] = enr["ca"]
    enr["prior_cgpa"] = np.where(enr["ca"] > 0, enr["cq"] / enr["ca"].where(enr["ca"] > 0, 1), np.nan)
    enr["load_credits"] = enr.groupby(["student_id", "sem_order"])["credit_hours"].transform("sum")
    enr["fail"] = (enr["result"] == "F").astype(int)
    return enr


def feature_columns(scenario: str) -> list:
    spec = SCENARIOS[scenario]
    return spec["numeric"] + spec["categorical"]


def build_preprocessor(scenario: str) -> ColumnTransformer:
    """Impute (a missing CGPA means a first-semester student), scale numbers, one-hot encode categories."""
    spec = SCENARIOS[scenario]
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([("num", numeric, spec["numeric"]), ("cat", categorical, spec["categorical"])])


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["attempted"] > 0].reset_index(drop=True)


def prepare_classification(frame: pd.DataFrame, scenario: str):
    """(X, y, groups): y = 1 for a failed course; groups = student ids."""
    f = _labelled(frame)
    return f[feature_columns(scenario)], f["fail"].to_numpy(int), f["student_id"].to_numpy()


def prepare_regression(frame: pd.DataFrame, scenario: str):
    f = _labelled(frame)
    return f[feature_columns(scenario)], f["mark"].to_numpy(float), f["student_id"].to_numpy()


def records_to_frame(records: list, scenario: str) -> pd.DataFrame:
    """Feature frame from dicts; missing fields become NaN (imputed)."""
    frame = pd.DataFrame([{k: (np.nan if v in (None, "") else v) for k, v in r.items()} for r in records])
    for col in feature_columns(scenario):
        if col not in frame.columns:
            frame[col] = np.nan
    for col in SCENARIOS[scenario]["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame[feature_columns(scenario)]
