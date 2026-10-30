"""
Data loader for the Academic Performance Monitoring & Prediction System.
Handles reading, writing, and computing statistics from the student CSV file.
"""

import os
import shutil
import pandas as pd
import numpy as np
from flask import current_app


SUBJECT_COLS = ["math", "science", "english", "history", "cs"]


def _get_data_file():
    data_file = current_app.config["DATA_FILE"]
    sample_file = current_app.config["SAMPLE_DATA_FILE"]
    # Bootstrap from sample data if main file doesn't exist yet
    if not os.path.exists(data_file) and os.path.exists(sample_file):
        shutil.copy(sample_file, data_file)
    return data_file


def load_students() -> pd.DataFrame:
    """Load student records from CSV, returning an empty DataFrame if file not found."""
    try:
        path = _get_data_file()
        df = pd.read_csv(path)
        df = _compute_derived_columns(df)
        return df
    except Exception:
        return pd.DataFrame()


def save_students(df: pd.DataFrame) -> None:
    """Persist the student DataFrame back to the CSV file."""
    path = _get_data_file()
    # Drop derived columns before saving to avoid duplication
    cols_to_drop = [c for c in ["avg_score", "gpa", "at_risk", "grade_letter"] if c in df.columns]
    df.drop(columns=cols_to_drop).to_csv(path, index=False)


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add avg_score, gpa, grade_letter, and at_risk derived columns."""
    present = [c for c in SUBJECT_COLS if c in df.columns]
    if present:
        df["avg_score"] = df[present].mean(axis=1).round(2)
    if "avg_score" in df.columns:
        df["gpa"] = (df["avg_score"] / 100 * 4.0).round(2)
        df["grade_letter"] = df["avg_score"].apply(_score_to_grade)
    if "attendance_pct" in df.columns and "avg_score" in df.columns:
        df["at_risk"] = (
            (df["avg_score"] < 50) | (df["attendance_pct"] < 60)
        ).astype(int)
    # Baseline for grade improvement/decline. If the column is absent (older CSV)
    # or missing for a newly added student, fall back to the current average so
    # improvement is treated as 0 (stable) rather than NaN.
    if "avg_score" in df.columns:
        if "previous_term_avg" not in df.columns:
            df["previous_term_avg"] = df["avg_score"]
        else:
            df["previous_term_avg"] = df["previous_term_avg"].fillna(df["avg_score"])
    return df


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


def get_student_by_id(student_id: int) -> dict | None:
    """Return a single student record as a dict, or None if not found."""
    df = load_students()
    if df.empty:
        return None
    row = df[df["student_id"] == int(student_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def add_student(data: dict) -> int:
    """Add a new student record and return the new student_id."""
    df = load_students()
    new_id = int(df["student_id"].max()) + 1 if not df.empty else 1
    data["student_id"] = new_id
    # Drop derived columns from new record before appending
    for col in ["avg_score", "gpa", "at_risk", "grade_letter"]:
        data.pop(col, None)
    new_row = pd.DataFrame([data])
    persistent_cols = [c for c in df.columns if c not in ["avg_score", "gpa", "at_risk", "grade_letter"]]
    base_df = df[persistent_cols] if not df.empty else pd.DataFrame()
    combined = pd.concat([base_df, new_row], ignore_index=True)
    save_students(combined)
    return new_id


def update_student(student_id: int, data: dict) -> bool:
    """Update an existing student record. Returns True on success."""
    df = load_students()
    persistent_cols = [c for c in df.columns if c not in ["avg_score", "gpa", "at_risk", "grade_letter"]]
    base_df = df[persistent_cols].copy()
    idx = base_df[base_df["student_id"] == int(student_id)].index
    if idx.empty:
        return False
    for key, val in data.items():
        if key in base_df.columns:
            base_df.at[idx[0], key] = val
    save_students(base_df)
    return True


def delete_student(student_id: int) -> bool:
    """Delete a student record. Returns True on success."""
    df = load_students()
    persistent_cols = [c for c in df.columns if c not in ["avg_score", "gpa", "at_risk", "grade_letter"]]
    base_df = df[persistent_cols].copy()
    before = len(base_df)
    base_df = base_df[base_df["student_id"] != int(student_id)]
    if len(base_df) == before:
        return False
    save_students(base_df)
    return True


def get_dashboard_stats() -> dict:
    """Compute summary statistics for the dashboard."""
    df = load_students()
    if df.empty:
        return {}

    stats = {
        "total_students": len(df),
        "pass_count": int(df["pass_fail"].sum()) if "pass_fail" in df.columns else 0,
        "fail_count": int((df["pass_fail"] == 0).sum()) if "pass_fail" in df.columns else 0,
        "avg_gpa": round(float(df["gpa"].mean()), 2) if "gpa" in df.columns else 0,
        "at_risk_count": int(df["at_risk"].sum()) if "at_risk" in df.columns else 0,
        "avg_attendance": round(float(df["attendance_pct"].mean()), 1) if "attendance_pct" in df.columns else 0,
    }
    stats["pass_rate"] = round(stats["pass_count"] / stats["total_students"] * 100, 1) if stats["total_students"] else 0

    # Grade distribution
    if "grade_letter" in df.columns:
        grade_dist = df["grade_letter"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
        stats["grade_distribution"] = grade_dist.to_dict()

    # Subject averages
    present_subjects = [c for c in SUBJECT_COLS if c in df.columns]
    if present_subjects:
        stats["subject_averages"] = {s: round(float(df[s].mean()), 1) for s in present_subjects}

    # Monthly/cohort trend (simulate using study_hours bins)
    if "study_hours_per_week" in df.columns and "avg_score" in df.columns:
        bins = pd.cut(df["study_hours_per_week"], bins=5)
        trend = df.groupby(bins, observed=False)["avg_score"].mean().round(1)
        stats["study_trend_labels"] = [str(b) for b in trend.index]
        stats["study_trend_values"] = trend.tolist()

    # Class-level improvement indicators (Requirement 3)
    if "avg_score" in df.columns and "previous_term_avg" in df.columns:
        improvement = df["avg_score"] - df["previous_term_avg"]
        stats["avg_improvement"] = round(float(improvement.mean()), 2)
        stats["improving_count"] = int((improvement > 2).sum())
        stats["declining_count"] = int((improvement < -2).sum())

    return stats


def get_analytics_data() -> dict:
    """Compute analytics data for charts."""
    df = load_students()
    if df.empty:
        return {}

    data = {}
    present_subjects = [c for c in SUBJECT_COLS if c in df.columns]

    # Top and bottom performers
    if "avg_score" in df.columns and "name" in df.columns:
        top = df.nlargest(5, "avg_score")[["name", "avg_score", "grade_letter"]].to_dict("records")
        bottom = df.nsmallest(5, "avg_score")[["name", "avg_score", "grade_letter"]].to_dict("records")
        data["top_performers"] = top
        data["bottom_performers"] = bottom

    # Subject breakdown by gender
    if "gender" in df.columns and present_subjects:
        gender_group = df.groupby("gender")[present_subjects].mean().round(1)
        data["gender_labels"] = gender_group.index.tolist()
        data["gender_subject_data"] = {s: gender_group[s].tolist() for s in present_subjects}
        data["subject_names"] = present_subjects

    # GPA distribution histogram
    if "gpa" in df.columns:
        counts, edges = np.histogram(df["gpa"].dropna(), bins=8, range=(0, 4))
        data["gpa_hist_labels"] = [f"{e:.1f}-{edges[i+1]:.1f}" for i, e in enumerate(edges[:-1])]
        data["gpa_hist_values"] = counts.tolist()

    # Attendance vs avg_score scatter (sample 50)
    if "attendance_pct" in df.columns and "avg_score" in df.columns:
        sample = df[["attendance_pct", "avg_score", "name"]].dropna().head(50)
        data["scatter_attendance"] = sample["attendance_pct"].tolist()
        data["scatter_score"] = sample["avg_score"].tolist()
        data["scatter_names"] = sample["name"].tolist()

    # Pass/fail by subject threshold
    if present_subjects and "pass_fail" in df.columns:
        data["subject_pass_rates"] = {
            s: round(float((df[s] >= 50).mean() * 100), 1) for s in present_subjects
        }

    # Class-level improvement indicators (Requirement 3)
    if "avg_score" in df.columns and "previous_term_avg" in df.columns:
        improvement = df["avg_score"] - df["previous_term_avg"]
        data["avg_improvement"] = round(float(improvement.mean()), 2)
        data["improving_count"] = int((improvement > 2).sum())
        data["declining_count"] = int((improvement < -2).sum())

    return data
