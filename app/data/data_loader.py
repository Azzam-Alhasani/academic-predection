"""
Data layer: three CSV tables shaped like a university transcript system.

    students.csv     student_id, name, gender, major, entry_semester
    courses.csv      course_code, course_name, major, credit_hours, level
    enrollments.csv  enrollment_id, student_id, semester, course_code, mark, attendance, status

status is Completed, Withdrawn (W) or In progress. Everything else (grade, points, GPA, standing,
attempt numbers, course difficulty) is calculated here from marks and credit hours using
app/grading.py, the way the transcript does it.

The tables returned by load_tables() are cached and must be treated as read-only.
"""

import os
import shutil
import threading

import numpy as np
import pandas as pd
from flask import current_app

from app import grading

FILES = ("students.csv", "courses.csv", "enrollments.csv")
STUDENT_COLUMNS = ["student_id", "name", "gender", "major", "entry_semester"]
ENROLLMENT_COLUMNS = ["enrollment_id", "student_id", "semester", "course_code", "mark", "attendance", "status"]
STATUSES = ("Completed", "Withdrawn", "In progress")
TERM_ORDER = {"Spring": 1, "Summer": 2, "Fall": 3}

_write_lock = threading.RLock()
_cache = {"key": None, "tables": None}
_cache_lock = threading.Lock()


class Tables:
    def __init__(self, students, courses, enrollments, summary):
        self.students, self.courses, self.enrollments, self.summary = students, courses, enrollments, summary

    @property
    def empty(self):
        return self.students.empty or self.enrollments.empty

    @property
    def current_semester(self):
        progress = self.enrollments[self.enrollments["status"] == "In progress"]
        pool = progress if not progress.empty else self.enrollments
        return None if pool.empty else max(pool["semester"], key=semester_order)


def semester_order(label: str) -> int:
    try:
        year, term = str(label).split()
        return int(year) * 10 + TERM_ORDER.get(term, 0)
    except ValueError:
        return 0


def _paths():
    data_dir, sample_dir = current_app.config["DATA_DIR"], current_app.config["SAMPLE_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    for name in FILES:                               # first start: copy the sample tables
        target = os.path.join(data_dir, name)
        if not os.path.exists(target) and os.path.exists(os.path.join(sample_dir, name)):
            shutil.copy(os.path.join(sample_dir, name), target)
    return {name: os.path.join(data_dir, name) for name in FILES}


# ── loading + calculations ───────────────────────────────────────────────────

def load_tables() -> Tables:
    """All tables with derived columns; empty tables if the files are missing."""
    try:
        paths = _paths()
        key = tuple((p, os.path.getmtime(p)) for p in paths.values())
    except (OSError, KeyError):
        return Tables(*(pd.DataFrame() for _ in range(4)))
    with _cache_lock:
        if _cache["key"] == key:
            return _cache["tables"]
        students = pd.read_csv(paths["students.csv"])
        courses = pd.read_csv(paths["courses.csv"])
        enrollments = pd.read_csv(paths["enrollments.csv"])
        tables = _derive(students, courses, enrollments)
        _cache.update(key=key, tables=tables)
        return tables


def _derive(students, courses, enrollments) -> Tables:
    enr = enrollments.merge(courses[["course_code", "course_name", "credit_hours", "level", "major"]]
                            .rename(columns={"major": "course_major"}), on="course_code", how="left")
    enr["mark"] = pd.to_numeric(enr["mark"], errors="coerce")
    enr["attendance"] = pd.to_numeric(enr["attendance"], errors="coerce")
    enr["credit_hours"] = enr["credit_hours"].fillna(3)
    enr["sem_order"] = enr["semester"].map(semester_order)
    done = enr["status"].eq("Completed") & enr["mark"].notna()

    letters = [grading.letter_for(m) if d else None for m, d in zip(enr["mark"], done)]
    enr["grade"] = np.where(done, letters, np.where(enr["status"].eq("Withdrawn"), "W", ""))
    enr["points"] = [grading.POINTS.get(g) if d else np.nan for g, d in zip(enr["grade"], done)]
    enr["result"] = np.where(done, np.where(enr["points"] >= grading.PASS_POINTS, "P", "F"),
                             np.where(enr["status"].eq("Withdrawn"), "W", "IP"))
    enr["attempted"] = np.where(done, enr["credit_hours"], 0.0)
    enr["earned"] = np.where(enr["result"].eq("P"), enr["credit_hours"], 0.0)
    enr["quality"] = np.where(done, enr["credit_hours"] * enr["points"].fillna(0), 0.0)
    enr = enr.sort_values(["student_id", "sem_order", "enrollment_id"]).reset_index(drop=True)
    enr["attempt"] = enr.groupby(["student_id", "course_code"]).cumcount() + 1
    enr["is_latest"] = ~enr.duplicated(["student_id", "course_code"], keep="last")
    enr["attendance_band"] = [grading.attendance_band(a) for a in enr["attendance"]]
    return Tables(students.copy(), courses.copy(), enr, _summarise(students, enr))


def _summarise(students, enr) -> pd.DataFrame:
    """One row per student: credits, CGPA, LCGPA, standing, current load, risk flag."""
    base = students.set_index("student_id")
    done = enr[enr["attempted"] > 0]
    total = done.groupby("student_id").agg(attempted=("attempted", "sum"), earned=("earned", "sum"),
                                           quality=("quality", "sum"))
    latest = done[done["is_latest"]].groupby("student_id").agg(l_attempted=("attempted", "sum"), l_quality=("quality", "sum"))
    cur = enr[enr["status"] == "In progress"].groupby("student_id").agg(
        current_credits=("credit_hours", "sum"), current_courses=("enrollment_id", "size"))
    extra = enr.groupby("student_id").agg(withdrawals=("result", lambda r: int((r == "W").sum())),
                                          repeats=("attempt", lambda a: int((a > 1).sum())),
                                          semesters=("semester", "nunique"))
    out = base.join(total).join(latest).join(cur).join(extra)
    for col in ("attempted", "earned", "quality", "l_attempted", "l_quality", "current_credits", "current_courses",
                "withdrawals", "repeats", "semesters"):
        out[col] = out[col].fillna(0)
    out["cgpa"] = np.where(out["attempted"] > 0, (out["quality"] / out["attempted"].where(out["attempted"] > 0, 1)).round(2), np.nan)
    out["lcgpa"] = np.where(out["l_attempted"] > 0, (out["l_quality"] / out["l_attempted"].where(out["l_attempted"] > 0, 1)).round(2), np.nan)
    out["standing"] = [grading.standing(None if pd.isna(c) else c) for c in out["cgpa"]]
    out["at_risk"] = (out["cgpa"].notna() & (out["cgpa"] < grading.GOOD_STANDING_CGPA)).astype(int)
    out["load_status"] = [grading.load_status(c) if c else "-" for c in out["current_credits"]]
    return out.reset_index()


# ── transcript for one student ───────────────────────────────────────────────

def get_student(student_id: int) -> dict | None:
    t = load_tables()
    row = t.summary[t.summary["student_id"] == int(student_id)] if not t.summary.empty else t.summary
    if row.empty:
        return None
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.iloc[0].to_dict().items()}


def transcript(student_id: int) -> list[dict]:
    """Semesters in order, each with its courses and SGPA / LCGPA / CGPA to date (like the printed transcript)."""
    t = load_tables()
    enr = t.enrollments[t.enrollments["student_id"] == int(student_id)]
    out, cum_att, cum_q = [], 0.0, 0.0
    seen = {}                                          # course -> (attempted, quality) of its latest attempt so far
    for semester, group in enr.groupby("sem_order", sort=True):
        att, qual, earned = float(group["attempted"].sum()), float(group["quality"].sum()), float(group["earned"].sum())
        for _, r in group.iterrows():
            if r["attempted"] > 0:
                seen[r["course_code"]] = (r["attempted"], r["quality"])
        cum_att, cum_q = cum_att + att, cum_q + qual
        l_att = sum(a for a, _ in seen.values()); l_q = sum(q for _, q in seen.values())
        credits = float(group.loc[group["status"] != "Withdrawn", "credit_hours"].sum())
        out.append({
            "semester": group["semester"].iloc[0], "in_progress": bool((group["status"] == "In progress").any()),
            "courses": group.to_dict("records"), "attempted": att, "earned": earned, "quality": round(qual, 1),
            "sgpa": round(qual / att, 2) if att else None,
            "lcgpa": round(l_q / l_att, 2) if l_att else None, "cgpa": round(cum_q / cum_att, 2) if cum_att else None,
            "cum_attempted": cum_att, "load": grading.load_status(credits) if credits else "-", "credits": credits,
        })
    return out


# ── course difficulty (measured from the marks) ──────────────────────────────

def course_stats(min_n: int = 15) -> pd.DataFrame:
    """
    Per course: attempts, mean mark, fail rate, withdrawal rate and a difficulty index.

        difficulty = average over attempts of (student's mean mark in their OTHER courses - mark here)

    A positive index means students score that many marks LOWER here than in their other courses.
    It compares each student with themselves, so a course is not "hard" just because weaker students take it.
    """
    t = load_tables()
    if t.empty:
        return pd.DataFrame()
    done = t.enrollments[t.enrollments["attempted"] > 0].copy()
    g = done.groupby("student_id")["mark"]
    n, s = g.transform("size"), g.transform("sum")
    done = done[n > 1].copy()
    done["other_mean"] = (s[done.index] - done["mark"]) / (n[done.index] - 1)
    done["gap"] = done["other_mean"] - done["mark"]
    stats = done.groupby("course_code").agg(attempts=("mark", "size"), mean_mark=("mark", "mean"),
                                            fail_rate=("result", lambda r: (r == "F").mean()),
                                            difficulty=("gap", "mean")).reset_index()
    wd = t.enrollments[t.enrollments["result"].isin(["W", "P", "F"])].groupby("course_code")["result"].agg(
        lambda r: (r == "W").mean()).rename("withdraw_rate")
    stats = stats.merge(wd, on="course_code", how="left").merge(
        t.courses[["course_code", "course_name", "major", "credit_hours", "level"]], on="course_code")
    stats = stats[stats["attempts"] >= min_n].copy()
    if len(stats) >= 3:
        hi, lo = stats["difficulty"].quantile([2 / 3, 1 / 3])
        stats["tier"] = np.where(stats["difficulty"] >= hi, "Hard", np.where(stats["difficulty"] <= lo, "Easy", "Medium"))
    else:
        stats["tier"] = "Medium"
    return stats.round({"mean_mark": 1, "fail_rate": 3, "difficulty": 1, "withdraw_rate": 3}).sort_values("difficulty", ascending=False)


# ── writing ──────────────────────────────────────────────────────────────────

def _write(name: str, df: pd.DataFrame) -> None:
    df.to_csv(_paths()[name], index=False)


def _raw(name: str) -> pd.DataFrame:
    return pd.read_csv(_paths()[name])


def add_student(data: dict) -> int:
    with _write_lock:
        df = _raw("students.csv")
        new_id = int(df["student_id"].max()) + 1 if not df.empty else 1
        row = {"student_id": new_id, "name": data.get("name") or f"Student {new_id:04d}", "gender": data["gender"],
               "major": data["major"], "entry_semester": data["entry_semester"]}
        _write("students.csv", pd.concat([df, pd.DataFrame([row])], ignore_index=True)[STUDENT_COLUMNS])
        return new_id


def update_student(student_id: int, data: dict) -> bool:
    with _write_lock:
        df = _raw("students.csv")
        idx = df.index[df["student_id"] == int(student_id)]
        if idx.empty:
            return False
        for key in ("name", "gender", "major", "entry_semester"):
            if key in data and data[key] not in (None, ""):
                df.at[idx[0], key] = data[key]
        _write("students.csv", df)
        return True


def delete_student(student_id: int) -> bool:
    with _write_lock:
        df = _raw("students.csv")
        rest = df[df["student_id"] != int(student_id)]
        if len(rest) == len(df):
            return False
        _write("students.csv", rest)
        enr = _raw("enrollments.csv")
        _write("enrollments.csv", enr[enr["student_id"] != int(student_id)])
        return True


def add_enrollment(data: dict) -> int:
    with _write_lock:
        df = _raw("enrollments.csv")
        new_id = int(df["enrollment_id"].max()) + 1 if not df.empty else 1
        row = {"enrollment_id": new_id, **{k: data.get(k) for k in ENROLLMENT_COLUMNS[1:]}}
        _write("enrollments.csv", pd.concat([df, pd.DataFrame([row])], ignore_index=True)[ENROLLMENT_COLUMNS])
        return new_id


def update_enrollment(enrollment_id: int, data: dict) -> bool:
    with _write_lock:
        df = _raw("enrollments.csv")
        idx = df.index[df["enrollment_id"] == int(enrollment_id)]
        if idx.empty:
            return False
        for key in ("mark", "attendance", "status"):
            if key in data:
                df.at[idx[0], key] = data[key]
        _write("enrollments.csv", df)
        return True


def delete_enrollment(enrollment_id: int) -> bool:
    with _write_lock:
        df = _raw("enrollments.csv")
        rest = df[df["enrollment_id"] != int(enrollment_id)]
        if len(rest) == len(df):
            return False
        _write("enrollments.csv", rest)
        return True
