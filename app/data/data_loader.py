"""
Data loader for the Academic Performance Monitoring & Prediction System.

Works on the real UCI "Student Performance" data (Math + Portuguese). Grades
G1, G2 and G3 are the first-period, second-period and final grades on a 0-20
scale; a student passes with G3 >= 10.
"""

import os
import shutil
import threading

import numpy as np
import pandas as pd
from flask import current_app

PASS_MARK = 10

# Serialises read-modify-write cycles on the CSV (the web server runs several threads).
_write_lock = threading.RLock()
GRADE_COLS = ["G1", "G2", "G3"]
DERIVED_COLS = ["final_score", "gpa", "grade_letter", "at_risk", "progress"]

# studytime is coded 1-4 in the original questionnaire.
STUDYTIME_LABELS = {1: "< 2 h", 2: "2-5 h", 3: "5-10 h", 4: "> 10 h"}


def _get_data_file():
    data_file = current_app.config["DATA_FILE"]
    sample_file = current_app.config["SAMPLE_DATA_FILE"]
    # Bootstrap the working copy from the pristine real dataset on first run.
    if not os.path.exists(data_file) and os.path.exists(sample_file):
        shutil.copy(sample_file, data_file)
    return data_file


def load_students() -> pd.DataFrame:
    """Load student records from CSV, returning an empty DataFrame if not found."""
    try:
        df = pd.read_csv(_get_data_file())
        return _compute_derived_columns(df)
    except Exception:
        return pd.DataFrame()


def save_students(df: pd.DataFrame) -> None:
    """Persist the student DataFrame back to the CSV file (derived columns excluded)."""
    path = _get_data_file()
    df.drop(columns=[c for c in DERIVED_COLS if c in df.columns]).to_csv(path, index=False)


def _persistent(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in DERIVED_COLS]].copy()


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add final_score (0-100), gpa (0-4), grade_letter, progress and at_risk."""
    for col in GRADE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in GRADE_COLS if c in df.columns]).copy()

    df["pass_fail"] = (df["G3"] >= PASS_MARK).astype(int)
    df["final_score"] = (df["G3"] / 20 * 100).round(1)
    df["gpa"] = (df["G3"] / 20 * 4.0).round(2)
    df["grade_letter"] = df["final_score"].apply(_score_to_grade)
    df["progress"] = (df["G3"] - df["G2"]).round(1)

    absences = pd.to_numeric(df.get("absences"), errors="coerce").fillna(0)
    failures = pd.to_numeric(df.get("failures"), errors="coerce").fillna(0)
    df["at_risk"] = ((df["G3"] < PASS_MARK) | (absences >= 10) | (failures >= 2)).astype(int)
    return df


def _score_to_grade(score: float) -> str:
    """Letter grade from a 0-100 score. D is the lowest passing grade (10/20 = 50)."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def get_student_by_id(student_id: int) -> dict | None:
    """Return a single student record as a dict, or None if not found."""
    df = load_students()
    if df.empty:
        return None
    row = df[df["student_id"] == int(student_id)]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    # Convert NaN (columns left blank for newly added students) to None for templates.
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}


def _add_student_unlocked(data: dict) -> int:
    """Add a new student record and return the new student_id."""
    df = load_students()
    base = _persistent(df) if not df.empty else pd.DataFrame()
    new_id = int(df["student_id"].max()) + 1 if not df.empty else 1
    data = {k: v for k, v in data.items() if k not in DERIVED_COLS}
    data["student_id"] = new_id
    if not str(data.get("name", "")).strip():
        data["name"] = f"Student {new_id:04d}"
    combined = pd.concat([base, pd.DataFrame([data])], ignore_index=True)
    save_students(combined)
    return new_id


def _update_student_unlocked(student_id: int, data: dict) -> bool:
    """Update an existing student record. Returns True on success."""
    df = load_students()
    if df.empty:
        return False
    base = _persistent(df)
    idx = base[base["student_id"] == int(student_id)].index
    if idx.empty:
        return False
    for key, val in data.items():
        if key in base.columns and key != "student_id":
            base.at[idx[0], key] = val
    save_students(base)
    return True


def _delete_student_unlocked(student_id: int) -> bool:
    """Delete a student record. Returns True on success."""
    df = load_students()
    if df.empty:
        return False
    base = _persistent(df)
    remaining = base[base["student_id"] != int(student_id)]
    if len(remaining) == len(base):
        return False
    save_students(remaining)
    return True


def add_student(data: dict) -> int:
    with _write_lock:
        return _add_student_unlocked(data)


def update_student(student_id: int, data: dict) -> bool:
    with _write_lock:
        return _update_student_unlocked(student_id, data)


def delete_student(student_id: int) -> bool:
    with _write_lock:
        return _delete_student_unlocked(student_id)


def _pass_rate(frame: pd.DataFrame) -> float:
    return round(float(frame["pass_fail"].mean() * 100), 1) if len(frame) else 0.0


def get_dashboard_stats() -> dict:
    """Compute summary statistics for the dashboard."""
    df = load_students()
    if df.empty:
        return {}

    total = len(df)
    stats = {
        "total_students": total,
        "pass_count": int(df["pass_fail"].sum()),
        "fail_count": int((df["pass_fail"] == 0).sum()),
        "avg_grade": round(float(df["G3"].mean()), 2),
        "at_risk_count": int(df["at_risk"].sum()),
        "avg_absences": round(float(pd.to_numeric(df["absences"], errors="coerce").mean()), 1),
    }
    stats["pass_rate"] = round(stats["pass_count"] / total * 100, 1)

    dist = df["grade_letter"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
    stats["grade_distribution"] = {k: int(v) for k, v in dist.items()}

    stats["period_averages"] = {g: round(float(df[g].mean()), 2) for g in GRADE_COLS}

    if "subject" in df.columns:
        by_subject = df.groupby("subject")
        stats["subject_summary"] = {
            name: {
                "students": int(len(g)),
                "avg_grade": round(float(g["G3"].mean()), 2),
                "pass_rate": _pass_rate(g),
            }
            for name, g in by_subject
        }

    if "studytime" in df.columns:
        grouped = df.groupby("studytime")["G3"].mean().round(2)
        stats["study_trend_labels"] = [STUDYTIME_LABELS.get(int(k), str(k)) for k in grouped.index]
        stats["study_trend_values"] = grouped.tolist()

    stats["avg_improvement"] = round(float(df["progress"].mean()), 2)
    stats["improving_count"] = int((df["progress"] >= 1).sum())
    stats["declining_count"] = int((df["progress"] <= -1).sum())
    return stats


def get_analytics_data() -> dict:
    """Compute analytics data for charts."""
    df = load_students()
    if df.empty:
        return {}

    data = {}
    cols = ["name", "subject", "G3", "grade_letter"]
    data["top_performers"] = df.nlargest(5, ["G3", "G2"])[cols].to_dict("records")
    data["bottom_performers"] = df.nsmallest(5, ["G3", "G2"])[cols].to_dict("records")

    # Final-grade histogram (0-20)
    counts = np.bincount(df["G3"].astype(int).clip(0, 20), minlength=21)
    data["grade_hist_labels"] = [str(i) for i in range(21)]
    data["grade_hist_values"] = counts.tolist()

    # Gender and subject comparisons
    for key, column in (("gender", "gender"), ("subject", "subject")):
        if column in df.columns:
            g = df.groupby(column)
            data[f"{key}_labels"] = [str(i) for i in g.size().index]
            data[f"{key}_avg_grade"] = g["G3"].mean().round(2).tolist()
            data[f"{key}_pass_rate"] = [_pass_rate(f) for _, f in g]

    # Pass rate by study time, past failures and higher-education plans
    if "studytime" in df.columns:
        g = df.groupby("studytime")
        data["studytime_labels"] = [STUDYTIME_LABELS.get(int(k), str(k)) for k in g.size().index]
        data["studytime_pass_rate"] = [_pass_rate(f) for _, f in g]
    if "failures" in df.columns:
        g = df.assign(_f=df["failures"].clip(upper=3)).groupby("_f")
        data["failures_labels"] = [("3+" if int(k) == 3 else str(int(k))) for k in g.size().index]
        data["failures_pass_rate"] = [_pass_rate(f) for _, f in g]
    if "higher" in df.columns:
        g = df.groupby("higher")
        data["higher_labels"] = [("Wants higher education" if k == "yes" else "Does not") for k in g.size().index]
        data["higher_pass_rate"] = [_pass_rate(f) for _, f in g]

    # Absences vs final grade scatter (fixed random sample keeps the page light)
    sample = df[["absences", "G3", "name"]].dropna()
    sample = sample.sample(n=min(250, len(sample)), random_state=1)
    data["scatter_absences"] = sample["absences"].tolist()
    data["scatter_grade"] = sample["G3"].tolist()
    data["scatter_names"] = sample["name"].tolist()

    data["avg_improvement"] = round(float(df["progress"].mean()), 2)
    data["improving_count"] = int((df["progress"] >= 1).sum())
    data["declining_count"] = int((df["progress"] <= -1).sum())
    return data
