"""
Explainable calculations and suggestions for each student.

Every number comes with its formula and the student's own values filled in, so the
student (or an examiner) can check it by hand. Marks are percentages (0-100); letters,
GPA and standing come from app/grading.py.
"""

import math

import numpy as np
import pandas as pd

from app import grading

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "good": 3}


def _num(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


# ── Class-level reference numbers ────────────────────────────────────────────

def build_class_stats(df: pd.DataFrame) -> dict:
    """Peer statistics per programme plus dropout rates used in the advice."""
    stats = {"programmes": {}, "overall": None, "rates": {}}
    if df.empty:
        return stats

    active = df[(df.s1_approved + df.s2_approved) > 0]

    def pack(frame):
        marks = np.sort(frame["avg_mark"].to_numpy(dtype=float))
        return {"n": int(len(marks)), "mean": float(marks.mean()), "std": float(marks.std()), "marks": marks}

    stats["overall"] = pack(active)
    for name, group in active.groupby("programme"):
        if len(group) >= 10:
            stats["programmes"][name] = pack(group)

    def rate(mask):
        sub = df[mask]
        return round(float(sub["dropout"].mean() * 100), 1) if len(sub) else None

    stats["rates"] = {
        "tuition_overdue": rate(df.tuition_paid == "no"), "tuition_paid": rate(df.tuition_paid == "yes"),
        "scholarship": rate(df.scholarship == "yes"), "no_scholarship": rate(df.scholarship == "no"),
        "debtor": rate(df.debtor == "yes"), "no_debt": rate(df.debtor == "no"),
        "completion_low": rate(df.completion.fillna(1) < 0.5), "completion_high": rate(df.completion.fillna(0) >= 0.75),
        "overall": rate(df.dropout >= 0),
    }
    return stats


def _peer(student, class_stats):
    """Percentile and z-score of the student's average mark among programme peers."""
    group = class_stats.get("programmes", {}).get(student.get("programme")) or class_stats.get("overall")
    if not group or group["std"] == 0:
        return None
    mark = _num(student.get("avg_mark"))
    marks = group["marks"]
    below = int(np.searchsorted(marks, mark, side="left"))
    equal = int(np.searchsorted(marks, mark, side="right")) - below
    percentile = (below + 0.5 * equal) / len(marks) * 100
    z = (mark - group["mean"]) / group["std"]
    return {"percentile": round(percentile, 1), "z": round(z, 2), "mean": round(group["mean"], 1),
            "std": round(group["std"], 1), "n": group["n"],
            "scope": student.get("programme") if student.get("programme") in class_stats["programmes"] else "all students"}


# ── The calculations (each with formula + working) ───────────────────────────

def calculate_student_indicators(student: dict, class_stats: dict | None = None,
                                 target_cgpa: float | None = None, next_courses: int | None = None) -> dict:
    class_stats = class_stats or {}
    e1, a1, m1 = _num(student.get("s1_enrolled")), _num(student.get("s1_approved")), _num(student.get("s1_mark"))
    e2, a2, m2 = _num(student.get("s2_enrolled")), _num(student.get("s2_approved")), _num(student.get("s2_mark"))
    enrolled, approved = e1 + e2, a1 + a2
    has_courses = enrolled > 0

    cgpa = student.get("cgpa")
    cgpa = None if cgpa is None or (isinstance(cgpa, float) and math.isnan(cgpa)) else float(cgpa)
    avg_mark = _num(student.get("avg_mark"))
    g1, g2, ga = grading.grade_for(m1), grading.grade_for(m2), grading.grade_for(avg_mark)

    calc = []   # list of {key, title, value, formula, working, meaning}

    def add(key, title, value, formula, working, meaning):
        calc.append({"key": key, "title": title, "value": value, "formula": formula,
                     "working": working, "meaning": meaning})

    if e1 > 0:
        gpa1 = grading.semester_gpa(m1, a1, e1)
        add("s1", "Semester 1 result", f"{m1:.1f}% - {g1['letter']} - GPA {gpa1:.2f}",
            "Semester GPA = GP(average mark) x courses passed / courses taken",
            f"GP({m1:.1f}) = {g1['points']} ; {g1['points']} x {int(a1)}/{int(e1)} = {gpa1:.2f}",
            f"{g1['description']}. A failed course counts as 0 grade points.")
    if e2 > 0:
        gpa2 = grading.semester_gpa(m2, a2, e2)
        add("s2", "Semester 2 result", f"{m2:.1f}% - {g2['letter']} - GPA {gpa2:.2f}",
            "Semester GPA = GP(average mark) x courses passed / courses taken",
            f"GP({m2:.1f}) = {g2['points']} ; {g2['points']} x {int(a2)}/{int(e2)} = {gpa2:.2f}",
            f"{g2['description']}.")
    if has_courses and cgpa is not None:
        add("cgpa", "Cumulative GPA (CGPA)", f"{cgpa:.2f} / 4.00",
            "CGPA = (GP1 x passed1 + GP2 x passed2) / (taken1 + taken2)",
            f"({g1['points']} x {int(a1)} + {g2['points']} x {int(a2)}) / ({int(e1)} + {int(e2)}) = {cgpa:.2f}",
            f"Academic standing: {grading.standing(cgpa)} (good standing from {grading.GOOD_STANDING_CGPA}).")
        rate = approved / enrolled
        add("completion", "Course completion rate", f"{rate * 100:.0f}%",
            "Completion = courses passed / courses taken",
            f"{int(approved)} / {int(enrolled)} = {rate:.2f}",
            f"The system treats {grading.MIN_COMPLETION_RATE * 100:.0f}% or more as healthy progress.")

    both = a1 > 0 and a2 > 0
    if both:
        delta = m2 - m1
        pct = delta / m1 * 100 if m1 else 0.0
        add("trend", "Mark trend", f"{delta:+.1f} marks ({pct:+.1f}%)",
            "Change = Mark(semester 2) - Mark(semester 1) ; % change = Change / Mark(semester 1) x 100",
            f"{m2:.1f} - {m1:.1f} = {delta:+.1f} ; {delta:+.1f} / {m1:.1f} x 100 = {pct:+.1f}%",
            "Improving" if delta >= 2 else "Declining" if delta <= -2 else "Stable")

    if approved > 0:
        nxt = grading.next_grade_up(avg_mark)
        buffer = avg_mark - grading.PASS_MARK
        add("buffer", "Safety margin above the pass mark", f"{buffer:+.1f} marks",
            f"Margin = average mark - pass mark ({grading.PASS_MARK:.0f})",
            f"{avg_mark:.1f} - {grading.PASS_MARK:.0f} = {buffer:+.1f}",
            "Average of the passed courses, weighted by how many were passed each semester.")
        if nxt:
            add("next_grade", f"Distance to the next grade ({nxt['letter']})", f"{nxt['gap']:.1f} marks",
                "Gap = lower bound of the next grade - current average mark",
                f"{nxt['mark']} - {avg_mark:.1f} = {nxt['gap']:.1f}",
                f"Your average is a {ga['letter']} ({ga['description']}).")

        peer = _peer({**student, "avg_mark": avg_mark}, class_stats)
        if peer:
            add("peer", "Position among your peers", f"Better than {peer['percentile']:.0f}% (z = {peer['z']:+.2f})",
                "Percentile = (students below + 0.5 x equal) / N x 100 ; z = (mark - mean) / std dev",
                f"Group: {peer['scope']} (N={peer['n']}), mean {peer['mean']}, std dev {peer['std']} ; "
                f"z = ({avg_mark:.1f} - {peer['mean']}) / {peer['std']} = {peer['z']:+.2f}",
                "z above 0 means above the group average.")

    # ---- planning: what is needed next semester --------------------------------
    plan = None
    if has_courses and cgpa is not None:
        typical = max(1, int(round(enrolled / (2 if e2 > 0 and e1 > 0 else 1))))
        n = int(next_courses) if next_courses else min(typical, 8)
        n = max(1, min(n, 12))
        to_pass, feasible = grading.courses_to_pass(enrolled, approved, n)
        add("to_pass", f"Courses to pass next semester ({n} planned)", f"{to_pass} of {n}",
            "k = ceil(0.75 x (taken + n) - passed)",
            f"ceil(0.75 x ({int(enrolled)} + {n}) - {int(approved)}) = {to_pass}",
            "Passing at least this many brings your completion rate to 75%."
            if feasible else "Even passing every course will not reach 75% in one semester; plan two semesters.")

        default_target = min(grading.MAX_POINTS, round((math.floor(cgpa * 2) + 1) / 2, 1))
        target = float(target_cgpa) if target_cgpa is not None else default_target
        target = max(0.0, min(grading.MAX_POINTS, target))
        needed, reachable = grading.required_semester_gpa(cgpa, enrolled, target, n)
        mark_needed = grading.mark_for_points(needed) if reachable else None
        working = (f"({target:.2f} x ({int(enrolled)} + {n}) - {cgpa:.2f} x {int(enrolled)}) / {n} = {needed:.2f}")
        if needed <= 0:
            meaning = "You already meet this target; keep passing your courses to hold it."
        elif not reachable:
            meaning = "Not reachable in one semester (it would need more than 4.00). Choose a lower target or plan two semesters."
        else:
            meaning = (f"Pass all {n} courses with an average of about {mark_needed:.0f}% "
                       f"({grading.letter_for(mark_needed)} or better).")
        add("planner", f"Goal planner: CGPA {target:.2f}", f"Semester GPA of {needed:.2f} needed",
            "Needed GPA = (target x (taken + n) - CGPA x taken) / n",
            working, meaning)
        plan = {"target": target, "courses": n, "needed_gpa": needed, "reachable": reachable,
                "mark_needed": mark_needed, "default_target": default_target}

        if both:
            projected = max(0.0, min(100.0, m2 + (m2 - m1)))
            proj_gpa = grading.points_for(projected)
            proj_cgpa = round((cgpa * enrolled + proj_gpa * n) / (enrolled + n), 2)
            add("projection", "If your trend continues", f"Next mark about {projected:.0f}% - CGPA {proj_cgpa:.2f}",
                "Next mark = Mark2 + (Mark2 - Mark1) ; new CGPA = (CGPA x taken + GP(next) x n) / (taken + n), all n courses passed",
                f"{m2:.1f} + ({m2:.1f} - {m1:.1f}) = {projected:.1f} ; "
                f"({cgpa:.2f} x {int(enrolled)} + {proj_gpa} x {n}) / ({int(enrolled)} + {n}) = {proj_cgpa:.2f}",
                "A simple straight-line projection, not a prediction: use it as a warning or a target.")

        per_sem = approved / (2 if e1 > 0 and e2 > 0 else 1)
        load = max(3, min(typical, math.ceil(per_sem + 1)))
        add("load", "Suggested course load", f"{load} courses",
            "Load = min(usual load, ceil(courses passed per semester + 1))",
            f"min({typical}, ceil({per_sem:.1f} + 1)) = {load}",
            "Taking slightly more than you usually pass is a stretch; far more risks failing several courses.")

    return {"calculations": calc, "plan": plan, "grade": ga, "cgpa": cgpa, "has_courses": has_courses,
            "scale": grading.scale_table()}


# ── Suggestions (rule based, each with the reason) ──────────────────────────

def recommend_student(student: dict, class_stats: dict | None = None, indicators: dict | None = None,
                      ml: dict | None = None) -> list[dict]:
    """Prioritised suggestions: [{priority, title, text, why}]."""
    class_stats = class_stats or {}
    ind = indicators or calculate_student_indicators(student, class_stats)
    calc = {c["key"]: c for c in ind["calculations"]}
    rates = class_stats.get("rates", {})
    out = []

    def add(priority, title, text, why):
        out.append({"priority": priority, "title": title, "text": text, "why": why})

    e1, a1, m1 = _num(student.get("s1_enrolled")), _num(student.get("s1_approved")), _num(student.get("s1_mark"))
    e2, a2, m2 = _num(student.get("s2_enrolled")), _num(student.get("s2_approved")), _num(student.get("s2_mark"))
    enrolled, approved = e1 + e2, a1 + a2

    if not ind["has_courses"]:
        add("medium", "No courses recorded yet",
            "There are no enrolled courses on record, so no GPA can be calculated. Register for courses and "
            "meet your academic advisor to plan your first semester.", "Courses taken = 0")
        return out

    cgpa = ind["cgpa"]
    rate = approved / enrolled
    if rate < 0.5:
        add("high", "You passed fewer than half of your courses",
            f"You passed {int(approved)} of {int(enrolled)} courses ({rate * 100:.0f}%). Talk to your advisor this week "
            f"and consider a lighter load next semester ({calc['load']['value'] if 'load' in calc else 'fewer courses'}). "
            + (f"In the data, {rates['completion_low']}% of students in this situation withdrew, against "
               f"{rates['completion_high']}% of those who passed 75% or more." if rates.get("completion_low") is not None else ""),
            f"Completion = {int(approved)}/{int(enrolled)} = {rate:.2f} (below 0.50)")
    elif rate < grading.MIN_COMPLETION_RATE and "to_pass" in calc:
        add("medium", "Raise your course completion rate",
            f"Your completion rate is {rate * 100:.0f}%. Pass {calc['to_pass']['value']} next semester to reach "
            f"{grading.MIN_COMPLETION_RATE * 100:.0f}%. Start with the courses you retook or missed.",
            calc["to_pass"]["working"])

    if cgpa is not None and cgpa < grading.WARNING_CGPA:
        add("high", "Your CGPA is in the probation range",
            f"CGPA {cgpa:.2f} is below {grading.WARNING_CGPA}. Book an academic-advising meeting and agree a recovery "
            "plan before registering for next semester.", calc["cgpa"]["working"])
    elif cgpa is not None and cgpa < grading.GOOD_STANDING_CGPA:
        add("medium", "Your CGPA is in the warning range",
            f"CGPA {cgpa:.2f} is below the good-standing line ({grading.GOOD_STANDING_CGPA}). Use the goal planner "
            "below to see what you need next semester.", calc["cgpa"]["working"])

    if "trend" in calc:
        delta = m2 - m1
        if delta <= -5:
            add("high", "Your marks are dropping", f"Your average fell {abs(delta):.1f} marks from semester 1 to 2. "
                "Find out what changed (workload, attendance, a specific subject) and ask for help early.",
                calc["trend"]["working"])
        elif delta <= -2:
            add("medium", "Your marks are slipping", f"Down {abs(delta):.1f} marks. Review your weakest courses and "
                "set a weekly revision slot.", calc["trend"]["working"])
        elif delta >= 2:
            add("good", "Your marks are improving", f"Up {delta:.1f} marks. Keep the routine that is working.",
                calc["trend"]["working"])

    if "next_grade" in calc:
        gap = grading.next_grade_up(_num(student.get("avg_mark")))
        if gap and gap["gap"] <= 3:
            add("low", f"You are close to a {gap['letter']}", f"Only {gap['gap']:.1f} marks short. Small gains in "
                "coursework and the final exam can move you up a band.", calc["next_grade"]["working"])

    if "peer" in calc:
        pct = float(calc["peer"]["value"].split("than ")[1].split("%")[0])
        if pct < 25:
            add("medium", "You are in the lower quarter of your group",
                "Join tutoring or a study group for your programme, and compare your notes with a student who scores higher.",
                calc["peer"]["working"])

    if student.get("tuition_paid") == "no":
        add("high", "Tuition fees are overdue",
            "Contact the finance office about a payment plan now. "
            + (f"In the data, {rates['tuition_overdue']}% of students with overdue fees withdrew, against "
               f"{rates['tuition_paid']}% with fees paid." if rates.get("tuition_overdue") is not None else ""),
            "tuition_paid = no")
    if student.get("debtor") == "yes":
        add("medium", "You have an outstanding debt", "Check what is owed and ask about instalments before it blocks registration.",
            "debtor = yes")
    if student.get("scholarship") != "yes" and (cgpa or 0) >= grading.GOOD_STANDING_CGPA:
        add("low", "Look into scholarships", "Your results are in good standing, so you may qualify for financial support. "
            "Ask the student affairs office which scholarships are open.", "scholarship = no, standing good")

    if ml and ml.get("p_dropout") is not None:
        p = ml["p_dropout"]
        if p >= 50:
            add("high", "The model flags a high withdrawal risk", f"Estimated risk {p:.0f}%. Please meet an advisor; "
                "this is an alert to act on, not a verdict.", f"Model probability = {p:.1f}% (>= 50%)")
        elif p >= 20:
            add("medium", "The model sees a moderate withdrawal risk", f"Estimated risk {p:.0f}%. Check the factors below "
                "and tackle the biggest one first.", f"Model probability = {p:.1f}% (20-50%)")

    if "planner" in calc and ind["plan"]:
        plan = ind["plan"]
        add("low", "Your goal for next semester", f"{calc['planner']['title']}: {calc['planner']['meaning']}",
            calc["planner"]["working"])

    if not out:
        add("good", "You are on track", "No warning signs in your record. Keep passing your courses and checking your marks.",
            "All indicators within normal range")
    out.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return out


def report_text(student: dict, indicators: dict, suggestions: list, ml: dict | None = None) -> str:
    """Plain-text version of the report (for copy / email)."""
    lines = [f"Academic progress report - {student.get('name')} ({student.get('programme')})", ""]
    for c in indicators["calculations"]:
        lines += [f"{c['title']}: {c['value']}", f"  Formula: {c['formula']}", f"  Working: {c['working']}", ""]
    if ml and ml.get("p_dropout") is not None:
        lines += [f"Estimated withdrawal risk: {ml['p_dropout']:.0f}% ({ml.get('level')})", ""]
    lines.append("Suggestions")
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. [{s['priority'].upper()}] {s['title']}: {s['text']}")
    return "\n".join(lines)


def get_all_recommendations(df: pd.DataFrame, class_stats: dict) -> list[dict]:
    """Top suggestions for each student in `df` (used by the staff overview)."""
    rows = []
    for record in df.to_dict("records"):
        record = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}
        ind = calculate_student_indicators(record, class_stats)
        sugg = recommend_student(record, class_stats, ind)
        rows.append({"student": record, "top": sugg[:3], "count": len(sugg)})
    return rows
