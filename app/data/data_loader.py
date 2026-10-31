"""
Data loader for the Academic Performance Monitoring & Prediction System.

Works on the real UCI "Predict Students' Dropout and Academic Success" data
(4,424 university students). Marks are percentages (0-100); letters, GPA, CGPA and
academic standing come from app/grading.py.
"""

import os
import shutil
import threading

import numpy as np
import pandas as pd
from flask import current_app

from app import grading

OUTCOMES = ["Graduate", "Enrolled", "Dropout"]
DERIVED_COLS = [
    "s1_gpa", "s2_gpa", "cgpa", "standing", "avg_mark", "letter", "completion",
    "mark_change", "dropout", "at_risk",
]

# Serialises read-modify-write cycles on the CSV (the web server runs several threads).
_write_lock = threading.RLock()


def _get_data_file():
    data_file = current_app.config["DATA_FILE"]
    sample_file = current_app.config["SAMPLE_DATA_FILE"]
    # Bootstrap the working copy from the pristine real dataset on first run.
    if not os.path.exists(data_file) and os.path.exists(sample_file):
        shutil.copy(sample_file, data_file)
    return data_file


def load_students() -> pd.DataFrame:
    """Load student records, returning an empty DataFrame if the file is missing."""
    try:
        return _compute_derived_columns(pd.read_csv(_get_data_file()))
    except Exception:
        return pd.DataFrame()


def save_students(df: pd.DataFrame) -> None:
    """Persist the student table (derived columns are never stored)."""
    df.drop(columns=[c for c in DERIVED_COLS if c in df.columns]).to_csv(_get_data_file(), index=False)


def _persistent(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in DERIVED_COLS]].copy()


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add GPA, CGPA, letter, standing, completion rate, trend and risk flags."""
    num_cols = ["s1_enrolled", "s1_approved", "s1_mark", "s2_enrolled", "s2_approved", "s2_mark"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["s1_gpa"] = [grading.semester_gpa(m, a, e) for m, a, e in zip(df.s1_mark, df.s1_approved, df.s1_enrolled)]
    df["s2_gpa"] = [grading.semester_gpa(m, a, e) for m, a, e in zip(df.s2_mark, df.s2_approved, df.s2_enrolled)]
    df["cgpa"] = [
        grading.cgpa(m1, a1, e1, m2, a2, e2)
        for m1, a1, e1, m2, a2, e2 in zip(
            df.s1_mark, df.s1_approved, df.s1_enrolled, df.s2_mark, df.s2_approved, df.s2_enrolled)
    ]
    df["s1_gpa"] = pd.to_numeric(df["s1_gpa"], errors="coerce")
    df["s2_gpa"] = pd.to_numeric(df["s2_gpa"], errors="coerce")
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")
    df["standing"] = [grading.standing(None if pd.isna(c) else c) for c in df["cgpa"]]

    approved = df.s1_approved + df.s2_approved
    enrolled = df.s1_enrolled + df.s2_enrolled
    weighted = df.s1_mark * df.s1_approved + df.s2_mark * df.s2_approved
    df["avg_mark"] = np.where(approved > 0, weighted / approved.where(approved > 0, 1), 0.0).round(1)
    df["letter"] = [grading.letter_for(m) for m in df["avg_mark"]]
    df["completion"] = np.where(enrolled > 0, approved / enrolled.where(enrolled > 0, 1), np.nan).round(3)

    both = (df.s1_approved > 0) & (df.s2_approved > 0)
    df["mark_change"] = np.where(both, (df.s2_mark - df.s1_mark).round(1), np.nan)

    df["dropout"] = (df["outcome"] == "Dropout").astype(int)
    cgpa_low = df["cgpa"].fillna(4.0) < grading.GOOD_STANDING_CGPA
    low_completion = df["completion"].fillna(1.0) < 0.5
    df["at_risk"] = ((cgpa_low | low_completion) & (enrolled > 0)).astype(int)
    return df


def get_student_by_id(student_id: int) -> dict | None:
    """Return a single student record as a dict, or None if not found."""
    df = load_students()
    if df.empty:
        return None
    row = df[df["student_id"] == int(student_id)]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    # Convert NaN to None for templates.
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}


def _add_student_unlocked(data: dict) -> int:
    df = load_students()
    base = _persistent(df) if not df.empty else pd.DataFrame()
    new_id = int(df["student_id"].max()) + 1 if not df.empty else 1
    data = {k: v for k, v in data.items() if k not in DERIVED_COLS}
    data["student_id"] = new_id
    if not str(data.get("name", "")).strip():
        data["name"] = f"Student {new_id:04d}"
    save_students(pd.concat([base, pd.DataFrame([data])], ignore_index=True))
    return new_id


def _update_student_unlocked(student_id: int, data: dict) -> bool:
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


# ── Aggregate statistics ─────────────────────────────────────────────────────

def _rate(frame: pd.DataFrame) -> float:
    return round(float(frame["dropout"].mean() * 100), 1) if len(frame) else 0.0


def _group_rates(df: pd.DataFrame, column: str, order=None, min_size: int = 1) -> dict:
    """Dropout rate, size and mean CGPA per group (groups smaller than min_size are skipped)."""
    groups = [(k, g) for k, g in df.groupby(column) if len(g) >= min_size]
    if order:
        groups.sort(key=lambda kg: order.index(kg[0]) if kg[0] in order else 99)
    return {
        "labels": [str(k) for k, _ in groups],
        "dropout_rate": [_rate(g) for _, g in groups],
        "avg_cgpa": [round(float(g["cgpa"].mean()), 2) for _, g in groups],
        "size": [int(len(g)) for _, g in groups],
    }


def get_dashboard_stats() -> dict:
    """Summary numbers and chart data for the dashboard."""
    df = load_students()
    if df.empty:
        return {}

    total = len(df)
    graded = df[(df.s1_approved + df.s2_approved) > 0]
    stats = {
        "total_students": total,
        "graduates": int((df["outcome"] == "Graduate").sum()),
        "enrolled": int((df["outcome"] == "Enrolled").sum()),
        "dropouts": int(df["dropout"].sum()),
        "dropout_rate": _rate(df),
        "avg_cgpa": round(float(df["cgpa"].mean()), 2),
        "probation": int((df["standing"] == "Academic probation").sum()),
        "at_risk_count": int(df["at_risk"].sum()),
        "avg_completion": round(float(df["completion"].mean() * 100), 1),
    }
    stats["outcome_counts"] = {o: int((df["outcome"] == o).sum()) for o in OUTCOMES}

    dist = graded["letter"].value_counts().reindex([b[1] for b in grading.GRADE_SCALE], fill_value=0)
    stats["letter_distribution"] = {k: int(v) for k, v in dist.items()}

    def _mean(mask, col):
        value = df.loc[mask, col].mean()
        return 0.0 if pd.isna(value) else round(float(value), 1)

    stats["semester_means"] = {
        outcome: [
            _mean((df.outcome == outcome) & (df.s1_approved > 0), "s1_mark"),
            _mean((df.outcome == outcome) & (df.s2_approved > 0), "s2_mark"),
        ]
        for outcome in OUTCOMES
    }
    prog = _group_rates(df, "programme", min_size=40)
    ranked = sorted(zip(prog["labels"], prog["dropout_rate"], prog["size"]), key=lambda t: -t[1])[:8]
    stats["programme_labels"] = [r[0] for r in ranked]
    stats["programme_dropout"] = [r[1] for r in ranked]
    stats["programme_size"] = [r[2] for r in ranked]
    stats["scale"] = grading.scale_table()
    return stats


def get_analytics_data() -> dict:
    """Chart data for the analytics page."""
    df = load_students()
    if df.empty:
        return {}

    data = {}
    data["programme"] = _group_rates(df, "programme", min_size=40)
    data["gender"] = _group_rates(df, "gender")
    data["scholarship"] = _group_rates(df, "scholarship", order=["no", "yes"])
    data["tuition_paid"] = _group_rates(df, "tuition_paid", order=["yes", "no"])
    data["debtor"] = _group_rates(df, "debtor", order=["no", "yes"])
    data["attendance"] = _group_rates(df, "attendance")

    age_order = ["<=19", "20-21", "22-25", "26-30", "31+"]
    ages = pd.cut(df["age"], [0, 19, 21, 25, 30, 120], labels=age_order)
    data["age"] = _group_rates(df.assign(age_band=ages.astype(str)), "age_band", order=age_order)

    comp_order = ["0-25%", "26-50%", "51-75%", "76-99%", "100%"]
    bands = pd.cut(df["completion"], [-0.01, 0.25, 0.5, 0.75, 0.999, 1.0], labels=comp_order)
    data["completion"] = _group_rates(
        df.assign(band=bands.astype(str)).dropna(subset=["completion"]), "band", order=comp_order)

    edges = np.arange(0, 4.5, 0.5)
    counts, _ = np.histogram(df["cgpa"].dropna(), bins=np.append(edges, 4.01))
    data["cgpa_hist_labels"] = [f"{a:.1f}-{a + 0.5:.1f}" if a < 4 else "4.0" for a in edges]
    data["cgpa_hist_values"] = counts.tolist()

    sample = df[(df.s1_approved > 0) & (df.s2_approved > 0)]
    sample = sample.sample(n=min(300, len(sample)), random_state=1)
    data["scatter_s1"] = sample["s1_mark"].tolist()
    data["scatter_s2"] = sample["s2_mark"].tolist()
    data["scatter_dropout"] = sample["dropout"].tolist()

    cols = ["student_id", "name", "programme", "cgpa", "standing"]
    ranked = df[df["cgpa"].notna() & ((df.s1_enrolled + df.s2_enrolled) > 0)]
    data["top"] = ranked.nlargest(5, ["cgpa", "avg_mark"])[cols].to_dict("records")
    data["bottom"] = ranked.nsmallest(5, ["cgpa", "avg_mark"])[cols].to_dict("records")
    return data
