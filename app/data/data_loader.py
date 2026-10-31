"""
Data loader for the Academic Performance Monitoring & Prediction System.

Works on the real UCI "Student Performance" data (Cortez & Silva, 2008). One record is one
student in one course with only the factors that matter: term 1 mark, term 2 mark, absences and
the final mark. Marks are percentages (0-100); letters, GPA and standing come from app/grading.py.
A record without a final mark is "In progress" (it can be scored, but not used for training).
"""

import os
import shutil
import threading

import numpy as np
import pandas as pd
from flask import current_app

from app import grading

RESULTS = ["Pass", "Fail", "In progress"]
DERIVED_COLS = [
    "t1_gpa", "t2_gpa", "gpa", "standing", "avg_mark", "letter", "trend", "absence_band",
    "result", "fail", "at_risk",
]
GPA_BANDS = ["0.0-0.4", "0.5-0.9", "1.0-1.4", "1.5-1.9", "2.0+"]

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
    """Add term GPAs, GPA, letter, standing, trend, absence band, result and risk flag."""
    df = df.copy()
    for col in ("t1_mark", "absences"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["t2_mark"] = pd.to_numeric(df["t2_mark"], errors="coerce")      # blank = term 2 not yet done
    df["final_mark"] = pd.to_numeric(df.get("final_mark"), errors="coerce")
    if "person_id" not in df.columns:
        df["person_id"] = range(1, len(df) + 1)

    df["t1_gpa"] = [grading.term_gpa(m) for m in df.t1_mark]
    df["t2_gpa"] = [np.nan if pd.isna(m) else grading.term_gpa(m) for m in df.t2_mark]
    df["gpa"] = [grading.gpa(a, b) for a, b in zip(df.t1_mark, df.t2_mark)]
    df["standing"] = [grading.standing(g) for g in df.gpa]
    df["avg_mark"] = [grading.average_mark(a, b) for a, b in zip(df.t1_mark, df.t2_mark)]
    df["letter"] = [grading.letter_for(m) for m in df.avg_mark]
    df["trend"] = (df.t2_mark - df.t1_mark).round(1)
    df["absence_band"] = [grading.absence_band(a) for a in df.absences]

    known = df["final_mark"].notna()
    df["fail"] = np.where(known, (df["final_mark"] < grading.PASS_MARK).astype(float), np.nan)
    df["result"] = np.where(~known, "In progress", np.where(df["final_mark"] < grading.PASS_MARK, "Fail", "Pass"))
    df["at_risk"] = (df.gpa < grading.GOOD_STANDING_GPA).astype(int)
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
    data["person_id"] = int(df["person_id"].max()) + 1 if not df.empty else 1
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

def _known(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["fail"].notna()]


def _fail_rate(frame: pd.DataFrame) -> float:
    return round(float(frame["fail"].mean() * 100), 1) if len(frame) else 0.0


def _group_rates(df: pd.DataFrame, column: str, order=None, min_size: int = 1) -> dict:
    """Fail rate, size and mean GPA per group, over records whose final mark is known."""
    df = _known(df)
    groups = [(k, g) for k, g in df.groupby(column) if len(g) >= min_size]
    if order:
        groups.sort(key=lambda kg: order.index(kg[0]) if kg[0] in order else 99)
    return {
        "labels": [str(k) for k, _ in groups],
        "fail_rate": [_fail_rate(g) for _, g in groups],
        "avg_gpa": [round(float(g["gpa"].mean()), 2) for _, g in groups],
        "size": [int(len(g)) for _, g in groups],
    }


def _gpa_band(series: pd.Series) -> pd.Series:
    return pd.cut(series, [-0.01, 0.49, 0.99, 1.49, 1.99, 4.01], labels=GPA_BANDS).astype(str)


def get_dashboard_stats() -> dict:
    """Summary numbers and chart data for the dashboard."""
    df = load_students()
    if df.empty:
        return {}
    known = _known(df)
    stats = {
        "total_students": len(df),
        "distinct_students": int(df["person_id"].nunique()),
        "passed": int((df["result"] == "Pass").sum()),
        "failed": int((df["result"] == "Fail").sum()),
        "in_progress": int((df["result"] == "In progress").sum()),
        "fail_rate": _fail_rate(known),
        "avg_gpa": round(float(df["gpa"].mean()), 2),
        "avg_absences": round(float(df["absences"].mean()), 1),
        "at_risk_count": int(df["at_risk"].sum()),
    }
    stats["result_counts"] = {r: int((df["result"] == r).sum()) for r in RESULTS if (df["result"] == r).any()}
    dist = df["letter"].value_counts().reindex([b[1] for b in grading.GRADE_SCALE], fill_value=0)
    stats["letter_distribution"] = {k: int(v) for k, v in dist.items()}

    def mean_of(frame, col):
        value = frame[col].mean()
        return 0.0 if pd.isna(value) else round(float(value), 1)

    stats["term_means"] = {
        r: [mean_of(known[known.result == r], c) for c in ("t1_mark", "t2_mark", "final_mark")]
        for r in ("Pass", "Fail") if (known.result == r).any()
    }
    by_gpa = _group_rates(df.assign(band=_gpa_band(df["gpa"])), "band", order=GPA_BANDS)
    stats["gpa_band"] = by_gpa
    stats["course"] = _group_rates(df, "course")
    stats["scale"] = grading.scale_table()
    return stats


def get_analytics_data() -> dict:
    """Chart data for the analytics page."""
    df = load_students()
    if df.empty:
        return {}
    data = {}
    data["gpa_band"] = _group_rates(df.assign(band=_gpa_band(df["gpa"])), "band", order=GPA_BANDS)
    data["absences"] = _group_rates(df, "absence_band", order=[b[2] for b in grading.ABSENCE_BANDS])
    data["course"] = _group_rates(df, "course")
    trend_order = ["Dropping (-5 or more)", "Slipping (-2 to -5)", "Stable", "Improving (+2 or more)"]
    trend = pd.cut(df["trend"], [-101, -5, -2, 1.99, 101], labels=trend_order, right=False).astype(str)
    data["trend"] = _group_rates(df.assign(band=trend), "band", order=trend_order)
    t1_order = ["Under 40", "40-49", "50-59", "60-69", "70+"]
    t1 = pd.cut(df["t1_mark"], [-1, 39.99, 49.99, 59.99, 69.99, 101], labels=t1_order).astype(str)
    data["term1"] = _group_rates(df.assign(band=t1), "band", order=t1_order)

    edges = np.arange(0, 4.5, 0.5)
    counts, _ = np.histogram(df["gpa"], bins=np.append(edges, 4.01))
    data["gpa_hist_labels"] = [f"{a:.1f}-{a + 0.5:.1f}" if a < 4 else "4.0" for a in edges]
    data["gpa_hist_values"] = counts.tolist()

    sample = _known(df).sample(n=min(300, len(_known(df))), random_state=1)
    data["scatter_t1"] = sample["t1_mark"].tolist()
    data["scatter_final"] = sample["final_mark"].tolist()
    data["scatter_fail"] = sample["fail"].astype(int).tolist()

    cols = ["student_id", "name", "course", "gpa", "standing"]
    data["top"] = df.nlargest(5, ["gpa", "avg_mark"])[cols].to_dict("records")
    data["bottom"] = df.nsmallest(5, ["gpa", "avg_mark"])[cols].to_dict("records")
    return data
