"""Aggregate numbers and chart data for the dashboard and analytics pages (no personal data)."""

import numpy as np
import pandas as pd

from app import grading
from app.data import data_loader


def _group(df: pd.DataFrame, by: str, order=None, value="mark") -> dict:
    """Mean mark, fail rate and size per group of completed course attempts."""
    groups = [(k, g) for k, g in df.groupby(by)]
    if order:
        groups.sort(key=lambda kg: order.index(kg[0]) if kg[0] in order else 99)
    return {"labels": [str(k) for k, _ in groups],
            "mean_mark": [round(float(g["mark"].mean()), 1) for _, g in groups],
            "fail_rate": [round(float((g["result"] == "F").mean() * 100), 1) for _, g in groups],
            "size": [int(len(g)) for _, g in groups]}


def get_dashboard_stats() -> dict:
    t = data_loader.load_tables()
    if t.empty:
        return {}
    enr, summ = t.enrollments, t.summary
    done = enr[enr["attempted"] > 0]
    graded = summ[summ["cgpa"].notna()]
    letters = [g[1] for g in grading.GRADE_SCALE]
    dist = done["grade"].value_counts().reindex(letters, fill_value=0)
    by_major = graded.groupby("major")["cgpa"].mean().round(2)
    stats = {
        "students": int(len(t.students)), "courses": int(len(t.courses)), "majors": int(t.students["major"].nunique()),
        "attempts": int(len(done)), "current_semester": t.current_semester,
        "in_progress": int((enr["status"] == "In progress").sum()),
        "avg_cgpa": round(float(graded["cgpa"].mean()), 2),
        "good_standing": int((summ["standing"] == "Good standing").sum()),
        "warning": int((summ["standing"] == "Academic warning").sum()),
        "probation": int((summ["standing"] == "Academic probation").sum()),
        "passed": int((enr["result"] == "P").sum()), "failed": int((enr["result"] == "F").sum()),
        "withdrawn": int((enr["result"] == "W").sum()),
        "fail_rate": round(float((enr["result"] == "F").sum() / max(1, len(done)) * 100), 1),
        "grade_distribution": {k: int(v) for k, v in dist.items()},
        "cgpa_by_major": {"labels": list(by_major.index), "values": [float(v) for v in by_major.values]},
        "result_counts": {"Passed": int((enr["result"] == "P").sum()), "Failed": int((enr["result"] == "F").sum()),
                          "Withdrawn": int((enr["result"] == "W").sum())},
        "scale": grading.scale_table(),
    }
    cs = data_loader.course_stats()
    hardest = cs.head(8)
    stats["hardest"] = {"labels": [f"{n} ({m[:3]})" if m != "Common" else n for n, m in zip(hardest["course_name"], hardest["major"])],
                        "values": [float(v) for v in hardest["difficulty"]]}
    return stats


def get_analytics_data() -> dict:
    t = data_loader.load_tables()
    if t.empty:
        return {}
    enr = t.enrollments.merge(t.students[["student_id", "gender", "major"]], on="student_id", how="left")
    done = enr[enr["attempted"] > 0].copy()
    done["load_status"] = done.groupby(["student_id", "sem_order"])["credit_hours"].transform("sum").map(grading.load_status)
    data = {
        "attendance": _group(done, "attendance_band", grading.ATTENDANCE_ORDER),
        "major": _group(done, "major"), "gender": _group(done, "gender"),
        "load": _group(done, "load_status", ["Light Load", "Normal Load", "Overload"]),
        "level": _group(done.assign(level=done["level"].map(lambda v: f"Year {int(v)}" if v == v else "?")), "level"),
    }
    summ = t.summary[t.summary["cgpa"].notna()]
    data["cgpa_gender"] = {"labels": sorted(summ["gender"].unique()),
                           "values": [round(float(summ.loc[summ["gender"] == g, "cgpa"].mean()), 2) for g in sorted(summ["gender"].unique())]}
    edges = np.arange(0, 4.5, 0.5)
    counts, _ = np.histogram(summ["cgpa"], bins=np.append(edges, 4.01))
    data["cgpa_hist"] = {"labels": [f"{a:.1f}-{a + 0.5:.1f}" if a < 4 else "4.0" for a in edges], "values": counts.tolist()}
    sample = done.sample(n=min(300, len(done)), random_state=1)
    data["scatter"] = {"x": sample["attendance"].round(1).tolist(), "y": sample["mark"].tolist(),
                       "fail": (sample["result"] == "F").astype(int).tolist()}
    cs = data_loader.course_stats()
    cols = ["course_name", "major", "credit_hours", "attempts", "mean_mark", "fail_rate", "withdraw_rate", "difficulty", "tier"]
    data["hardest"] = cs.head(10)[cols].to_dict("records")
    data["easiest"] = cs.tail(6).iloc[::-1][cols].to_dict("records")
    data["tier_fail"] = {tier: round(float(cs.loc[cs["tier"] == tier, "fail_rate"].mean() * 100), 1) for tier in ("Hard", "Medium", "Easy")}
    data["tier_withdraw"] = {tier: round(float(cs.loc[cs["tier"] == tier, "withdraw_rate"].mean() * 100), 1) for tier in ("Hard", "Medium", "Easy")}
    return data
