"""
Simple, explainable indicators and rule-based advice for each student.
All grades are on the 0-20 scale of the UCI Student Performance data (pass = 10).
"""

import pandas as pd

PASS_MARK = 10
TREND_THRESHOLD = 1.0      # grade points between G2 and G3
HIGH_ABSENCES = 10
STUDYTIME_LABELS = {1: "under 2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: "over 10 hours"}


def _num(student, key, default=0.0) -> float:
    value = student.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(number) else number


def classify_trend(progress: float, threshold: float = TREND_THRESHOLD) -> str:
    if progress >= threshold:
        return "Improving"
    if progress <= -threshold:
        return "Declining"
    return "Stable"


def build_class_stats(df) -> dict:
    stats = {}
    for col in ("G1", "G2", "G3"):
        if col in df.columns:
            stats[f"{col}_mean"] = round(float(df[col].mean()), 2)
    if "absences" in df.columns:
        stats["absences_median"] = float(pd.to_numeric(df["absences"], errors="coerce").median())
    return stats


def get_risk_factors(student) -> list[str]:
    """Plain-language list of the factors that put a student at risk."""
    factors = []
    if _num(student, "G2") < PASS_MARK:
        factors.append("Second-period grade below the pass mark")
    if _num(student, "failures") >= 1:
        factors.append(f"{int(_num(student, 'failures'))} past class failure(s)")
    if _num(student, "absences") >= HIGH_ABSENCES:
        factors.append(f"{int(_num(student, 'absences'))} school absences")
    if _num(student, "studytime", 2) <= 1:
        factors.append("Weekly study time under 2 hours")
    if str(student.get("higher", "yes")).lower() == "no":
        factors.append("No plan to continue to higher education")
    return factors


def calculate_student_indicators(student: dict, class_stats: dict | None = None) -> dict:
    class_stats = class_stats or {}
    g1, g2, g3 = _num(student, "G1"), _num(student, "G2"), _num(student, "G3")
    progress = round(g3 - g2, 2)
    return {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "progress": progress,
        "trend": classify_trend(progress),
        "risk_factors": get_risk_factors(student),
        "vs_class": round(g3 - class_stats.get("G3_mean", g3), 2),
        "absences": int(_num(student, "absences")),
        "studytime": int(_num(student, "studytime", 2)),
        "failures": int(_num(student, "failures")),
        "at_risk": int(_num(student, "at_risk")),
    }


def recommend_student(student: dict, class_stats: dict | None = None) -> list[str]:
    class_stats = class_stats or {}
    ind = calculate_student_indicators(student, class_stats)
    tips = []

    if ind["trend"] == "Declining":
        tips.append(
            f"The final grade dropped {abs(ind['progress']):.0f} point(s) compared with the "
            "second period. Talk to the teacher to find out where the decline is coming from."
        )
    elif ind["trend"] == "Improving":
        tips.append(
            f"The final grade improved by {ind['progress']:.0f} point(s) since the second "
            "period. Keep using the same study routine."
        )
    else:
        tips.append("Grades are steady across the periods. Set a small goal to raise them a point or two.")

    if ind["g3"] < PASS_MARK:
        tips.append(
            f"The final grade ({ind['g3']:.0f}/20) is below the pass mark of {PASS_MARK}. "
            "Ask about extra classes, revision sessions or a retake plan."
        )
    elif ind["g2"] < PASS_MARK:
        tips.append("The second-period grade was below the pass mark. Early support makes the biggest difference.")

    if ind["failures"] >= 1:
        tips.append(
            f"{ind['failures']} past failure(s) is one of the strongest warning signs in the data. "
            "Review the topics that were failed before moving on."
        )

    if ind["absences"] >= HIGH_ABSENCES:
        tips.append(
            f"{ind['absences']} absences is high. Missing lessons makes catching up much harder, "
            "so aim to attend every class."
        )
    elif ind["absences"] >= 5:
        median = class_stats.get("absences_median")
        extra = f" (class median: {median:.0f})" if median is not None else ""
        tips.append(f"{ind['absences']} absences{extra}. Try to keep absences low.")

    if ind["studytime"] <= 1:
        tips.append(
            f"Weekly study time is {STUDYTIME_LABELS.get(ind['studytime'], 'low')}. "
            "Adding two or three focused hours a week usually helps."
        )

    if str(student.get("higher", "yes")).lower() == "no":
        tips.append("Talking about future study options with a counsellor can raise motivation.")

    if ind["at_risk"] and ind["g3"] >= PASS_MARK and not ind["risk_factors"]:
        tips.append("Currently passing, but keep an eye on attendance and past results.")

    return tips


def get_all_recommendations(df) -> list[dict]:
    class_stats = build_class_stats(df)
    rows = []
    for _, row in df.iterrows():
        student = row.to_dict()
        ind = calculate_student_indicators(student, class_stats)
        rows.append({
            "student_id": int(student.get("student_id")),
            "name": student.get("name"),
            "subject": student.get("subject"),
            "g1": ind["g1"],
            "g2": ind["g2"],
            "g3": ind["g3"],
            "progress": ind["progress"],
            "trend": ind["trend"],
            "risk_factors": ind["risk_factors"],
            "at_risk": ind["at_risk"],
            "recommendations": recommend_student(student, class_stats),
        })
    return rows
