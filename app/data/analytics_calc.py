SUBJECT_COLS = ["math", "science", "english", "history", "cs"]
WEAK_SUBJECT_THRESHOLD = 60
TREND_THRESHOLD = 2.0


def build_class_stats(df) -> dict:
    stats = {}
    present = [c for c in SUBJECT_COLS if c in df.columns]
    if present:
        stats["subject_means"] = {s: round(float(df[s].mean()), 1) for s in present}
    if "study_hours_per_week" in df.columns:
        stats["study_hours_median"] = round(float(df["study_hours_per_week"].median()), 1)
    if "avg_score" in df.columns:
        stats["avg_score_mean"] = round(float(df["avg_score"].mean()), 1)
    return stats


def compute_improvement(avg_score, previous_term_avg) -> float:
    return round(float(avg_score) - float(previous_term_avg), 2)


def classify_trend(improvement, threshold=TREND_THRESHOLD) -> str:
    if improvement > threshold:
        return "Improving"
    if improvement < -threshold:
        return "Declining"
    return "Stable"


def get_weak_subjects(student, threshold=WEAK_SUBJECT_THRESHOLD) -> list:
    return [s for s in SUBJECT_COLS if float(student.get(s, 0)) < threshold]


def get_strongest_subject(student) -> str | None:
    scored = {s: float(student.get(s, 0)) for s in SUBJECT_COLS if s in student}
    if not scored:
        return None
    return max(scored, key=scored.get)


def get_subject_vs_class(student, class_stats) -> dict:
    means = class_stats.get("subject_means", {})
    out = {}
    for s in SUBJECT_COLS:
        if s in means and s in student:
            out[s] = round(float(student[s]) - float(means[s]), 1)
    return out


def calculate_student_indicators(student: dict, class_stats: dict | None = None) -> dict:
    class_stats = class_stats or {}
    avg = float(student.get("avg_score", 0))
    prev = float(student.get("previous_term_avg", avg))

    improvement = compute_improvement(avg, prev)
    trend = classify_trend(improvement)
    weak = get_weak_subjects(student)
    strongest = get_strongest_subject(student)

    assignments = float(student.get("assignments_submitted", 0))
    attendance = float(student.get("attendance_pct", 0))
    study = float(student.get("study_hours_per_week", 0))

    return {
        "avg_score": round(avg, 2),
        "previous_term_avg": round(prev, 2),
        "improvement": improvement,
        "trend": trend,
        "weak_subjects": weak,
        "strongest_subject": strongest,
        "subject_vs_class": get_subject_vs_class(student, class_stats),
        "assignments_gap": int(max(0, 12 - assignments)),
        "attendance": attendance,
        "study_hours": study,
        "at_risk": int(student.get("at_risk", 0)),
    }


def recommend_student(student: dict, class_stats: dict | None = None) -> list[str]:
    class_stats = class_stats or {}
    ind = calculate_student_indicators(student, class_stats)
    tips = []

    if ind["trend"] == "Declining":
        tips.append(
            "Your average dropped "
            f"{abs(ind['improvement']):.1f} points compared to last term. "
            "Meet your instructor to review where the decline is coming from."
        )
    elif ind["trend"] == "Improving":
        tips.append(
            f"Great job — your average improved by {ind['improvement']:.1f} points. "
            "Keep using the same study routine."
        )
    else:
        tips.append(
            "Your average is steady compared to last term. "
            "Set a small goal to push it a few points higher this term."
        )

    if ind["weak_subjects"]:
        subjects = ", ".join(ind["weak_subjects"]).title()
        tips.append(
            f"Focus on your weakest subject(s): {subjects}. "
            "Spend extra time on practice questions and ask for help early."
        )

    if ind["attendance"] < 60:
        tips.append(
            f"Attendance is {ind['attendance']:.0f}%, below the 60% safe level. "
            "Attending classes more regularly is the single biggest improvement you can make."
        )
    elif ind["attendance"] < 75:
        tips.append(
            f"Attendance is {ind['attendance']:.0f}%. "
            "Try to attend at least 75% of classes to avoid falling behind."
        )

    if ind["assignments_gap"] > 0:
        tips.append(
            f"You have {ind['assignments_gap']} missing assignment(s) out of 12. "
            "Submitting all remaining work will improve both your grade and your understanding."
        )

    median = class_stats.get("study_hours_median")
    if median is not None and ind["study_hours"] < median:
        tips.append(
            f"Your study time ({ind['study_hours']:.1f} h/week) is below the class median "
            f"({median:.1f} h/week). Try adding 2-3 focused study hours per week."
        )

    if not tips:
        tips.append("Keep up the good work — you are on track.")

    return tips


def get_all_recommendations(df) -> list[dict]:
    class_stats = build_class_stats(df)
    rows = []
    for _, row in df.iterrows():
        student = row.to_dict()
        ind = calculate_student_indicators(student, class_stats)
        rows.append({
            "student_id": student.get("student_id"),
            "name": student.get("name"),
            "avg_score": ind["avg_score"],
            "previous_term_avg": ind["previous_term_avg"],
            "improvement": ind["improvement"],
            "trend": ind["trend"],
            "weak_subjects": ind["weak_subjects"],
            "strongest_subject": ind["strongest_subject"],
            "recommendations": recommend_student(student, class_stats),
        })
    return rows