"""
Explainable calculations and suggestions for each student.

Every number comes with its formula and the student's own values filled in, so the student (or an
examiner) can check it by hand. Marks are percentages (0-100); letters, GPA and standing come from
app/grading.py. A record is one student in one course: term 1 mark, term 2 mark (may be blank),
absences and, once known, the final mark.
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


def _has(value) -> bool:
    return value is not None and value == value and value != ""


# ── Class-level reference numbers ────────────────────────────────────────────

def build_class_stats(df: pd.DataFrame) -> dict:
    """Peer statistics per course plus the real fail rates quoted in the advice."""
    stats = {"courses": {}, "overall": None, "rates": {}}
    if df.empty:
        return stats

    def pack(frame):
        marks = np.sort(frame["avg_mark"].to_numpy(dtype=float))
        absences = np.sort(frame["absences"].to_numpy(dtype=float))
        return {"n": int(len(marks)), "mean": float(marks.mean()), "std": float(marks.std()),
                "marks": marks, "absences": absences}

    stats["overall"] = pack(df)
    for name, group in df.groupby("course"):
        if len(group) >= 10:
            stats["courses"][name] = pack(group)

    known = df[df["fail"].notna()]

    def rate(mask):
        sub = known[mask.loc[known.index]]
        return round(float(sub["fail"].mean() * 100), 1) if len(sub) >= 5 else None

    stats["rates"] = {
        "overall": round(float(known["fail"].mean() * 100), 1) if len(known) else None,
        "absence_band": {b[2]: rate(df["absence_band"] == b[2]) for b in grading.ABSENCE_BANDS},
        "t1_low": rate(df["t1_mark"] < 50), "t1_ok": rate(df["t1_mark"] >= 60),
        "drop": rate(df["trend"] <= -5), "no_drop": rate(df["trend"] > -2),
    }
    return stats


def _peer(student, class_stats, key="marks", value=None, mean_key="mean"):
    group = class_stats.get("courses", {}).get(student.get("course")) or class_stats.get("overall")
    if not group:
        return None
    values = group[key]
    value = _num(student.get("avg_mark")) if value is None else value
    below = int(np.searchsorted(values, value, side="left"))
    equal = int(np.searchsorted(values, value, side="right")) - below
    percentile = (below + 0.5 * equal) / len(values) * 100
    scope = student.get("course") if student.get("course") in class_stats.get("courses", {}) else "all students"
    out = {"percentile": round(percentile, 1), "n": group["n"], "scope": scope}
    if key == "marks":
        std = group["std"] or 1.0
        out.update(z=round((value - group["mean"]) / std, 2), mean=round(group["mean"], 1), std=round(std, 1))
    return out


# ── The calculations (each with formula + working) ───────────────────────────

def calculate_student_indicators(student: dict, class_stats: dict | None = None,
                                 target_average: float | None = None) -> dict:
    class_stats = class_stats or {}
    m1 = _num(student.get("t1_mark"))
    has2 = _has(student.get("t2_mark"))
    m2 = _num(student.get("t2_mark")) if has2 else None
    absences = int(_num(student.get("absences")))
    marks = [m1] + ([m2] if has2 else [])
    avg = grading.average_mark(*marks)
    gpa = grading.gpa(*marks)
    g1 = grading.grade_for(m1)
    g2 = grading.grade_for(m2) if has2 else None
    ga = grading.grade_for(avg)

    calc = []   # list of {key, title, value, formula, working, meaning}

    def add(key, title, value, formula, working, meaning):
        calc.append({"key": key, "title": title, "value": value, "formula": formula,
                     "working": working, "meaning": meaning})

    add("t1", "Term 1", f"{m1:.1f}% - {g1['letter']} - {g1['points']:.1f} points",
        "Term GPA = grade points of the term mark (scale on the dashboard)",
        f"{m1:.1f}% falls in {g1['letter']} = {g1['points']:.1f}", f"{g1['description']}.")
    if has2:
        add("t2", "Term 2", f"{m2:.1f}% - {g2['letter']} - {g2['points']:.1f} points",
            "Term GPA = grade points of the term mark", f"{m2:.1f}% falls in {g2['letter']} = {g2['points']:.1f}",
            f"{g2['description']}.")
    add("gpa", "GPA", f"{gpa:.2f} / 4.00",
        "GPA = average of the term GPAs",
        (f"({g1['points']:.1f} + {g2['points']:.1f}) / 2 = {gpa:.2f}" if has2 else f"{g1['points']:.1f} / 1 = {gpa:.2f}"),
        f"Academic standing: {grading.standing(gpa)} (good standing from {grading.GOOD_STANDING_GPA}).")
    add("avg", "Average mark", f"{avg:.1f}% ({ga['letter']})",
        "Average = (term 1 + term 2) / number of terms",
        (f"({m1:.1f} + {m2:.1f}) / 2 = {avg:.1f}" if has2 else f"{m1:.1f} / 1 = {avg:.1f}"),
        f"Equivalent grade: {ga['letter']} ({ga['description']}).")

    if has2:
        delta = m2 - m1
        pct = delta / m1 * 100 if m1 else 0.0
        add("trend", "Mark trend", f"{delta:+.1f} marks ({pct:+.1f}%)",
            "Change = term 2 - term 1 ; % change = Change / term 1 x 100",
            f"{m2:.1f} - {m1:.1f} = {delta:+.1f} ; {delta:+.1f} / {m1:.1f} x 100 = {pct:+.1f}%",
            "Improving" if delta >= 2 else "Dropping" if delta <= -2 else "Stable")

    buffer = avg - grading.PASS_MARK
    add("buffer", "Safety margin above the pass mark", f"{buffer:+.1f} marks",
        f"Margin = average mark - pass mark ({grading.PASS_MARK:.0f})", f"{avg:.1f} - {grading.PASS_MARK:.0f} = {buffer:+.1f}",
        "Negative means you are currently below the pass mark." if buffer < 0 else "Positive means you are above the pass mark.")
    nxt = grading.next_grade_up(avg)
    if nxt:
        add("next_grade", f"Distance to the next grade ({nxt['letter']})", f"{nxt['gap']:.1f} marks",
            "Gap = lower bound of the next grade - average mark", f"{nxt['mark']} - {avg:.1f} = {nxt['gap']:.1f}",
            f"Your average is a {ga['letter']}.")

    peer = _peer(student, class_stats, "marks", avg)
    if peer:
        add("peer", "Position among your peers", f"Better than {peer['percentile']:.0f}% (z = {peer['z']:+.2f})",
            "Percentile = (students below + 0.5 x equal) / N x 100 ; z = (mark - mean) / std dev",
            f"Group: {peer['scope']} (N={peer['n']}), mean {peer['mean']}, std dev {peer['std']} ; "
            f"z = ({avg:.1f} - {peer['mean']}) / {peer['std']} = {peer['z']:+.2f}", "z above 0 means above the group average.")

    band = grading.absence_band(absences)
    band_rate = class_stats.get("rates", {}).get("absence_band", {}).get(band)
    att = _peer(student, class_stats, "absences", absences)
    add("absences", "Absences", f"{absences} ({band})",
        "Compared with peers: percentile of absences (lower is better)",
        (f"{absences} absences ; fewer absences than {100 - att['percentile']:.0f}% of {att['scope']}" if att else f"{absences} absences")
        + (f" ; {band_rate}% of students in this band failed the course" if band_rate is not None else ""),
        "Every absence is a missed lesson. The link with failing is weak below 11 absences and clearer above it.")

    # ---- planning: what the final mark has to be -------------------------------
    plan = None
    default_target = float(nxt["mark"]) if nxt and avg >= grading.PASS_MARK else float(grading.PASS_MARK) if avg < grading.PASS_MARK else 100.0
    if has2:
        target = default_target if target_average is None else max(0.0, min(100.0, float(target_average)))
        needed, reachable = grading.required_final_mark(m1, m2, target)
        if needed <= 0:
            meaning = "You already meet this target; keep your marks up."
        elif not reachable:
            meaning = "Not reachable with one more mark (it would need more than 100). Choose a lower target."
        else:
            meaning = (f"Score about {needed:.0f}% in the final mark ({grading.letter_for(needed)} or better), "
                       "assuming all three marks count equally.")
        add("planner", f"Goal planner: average {target:.0f}%", f"Final mark of {needed:.1f}% needed",
            "Needed = 3 x target - term 1 - term 2", f"3 x {target:.0f} - {m1:.1f} - {m2:.1f} = {needed:.1f}", meaning)
        plan = {"target": target, "needed_mark": needed, "reachable": reachable, "default_target": default_target}

        projected = max(0.0, min(100.0, m2 + (m2 - m1)))
        proj_avg = round((m1 + m2 + projected) / 3, 1)
        add("projection", "If your trend continues", f"Final mark about {projected:.0f}% - average {proj_avg}%",
            "Next mark = term 2 + (term 2 - term 1) ; average = (term 1 + term 2 + next) / 3",
            f"{m2:.1f} + ({m2:.1f} - {m1:.1f}) = {projected:.1f} ; ({m1:.1f} + {m2:.1f} + {projected:.1f}) / 3 = {proj_avg}",
            "A straight-line projection, not a prediction: use it as a warning or a target.")

    return {"calculations": calc, "plan": plan, "grade": ga, "gpa": gpa, "standing": grading.standing(gpa), "avg_mark": avg, "has_term2": has2,
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

    m1 = _num(student.get("t1_mark"))
    has2 = ind["has_term2"]
    m2 = _num(student.get("t2_mark")) if has2 else None
    avg, gpa = ind["avg_mark"], ind["gpa"]
    absences = int(_num(student.get("absences")))

    if avg < grading.PASS_MARK:
        add("high", "Your average is below the pass mark",
            f"Your average is {avg:.1f}% (pass mark {grading.PASS_MARK:.0f}%). Meet your teacher this week, list the topics "
            "you lost marks on, and set a fixed revision slot for them. "
            + (f"In the data, {rates['t1_low']}% of students under 50% in term 1 failed, against {rates['t1_ok']}% of those with 60% or more."
               if rates.get("t1_low") is not None and rates.get("t1_ok") is not None else ""),
            calc["buffer"]["working"])
    elif avg < grading.PASS_MARK + 10:
        add("medium", "You are close to the pass mark",
            f"Your margin is only {avg - grading.PASS_MARK:.1f} marks. A weak final could pull you under, so protect the margin: "
            "revise your lowest-scoring topics first.", calc["buffer"]["working"])

    if has2:
        delta = m2 - m1
        if delta <= -5:
            add("high", "Your marks are dropping",
                f"Your mark fell {abs(delta):.1f} points from term 1 to term 2. Find out what changed (topic, workload, attendance) and ask for help early. "
                + (f"In the data, {rates['drop']}% of students whose mark fell 5 or more points failed, against {rates['no_drop']}% of the rest."
                   if rates.get("drop") is not None and rates.get("no_drop") is not None else ""),
                calc["trend"]["working"])
        elif delta <= -2:
            add("medium", "Your marks are slipping", f"Down {abs(delta):.1f} points. Review the topics from the last term and set a weekly revision slot.",
                calc["trend"]["working"])
        elif delta >= 2:
            add("good", "Your marks are improving", f"Up {delta:.1f} points. Keep the routine that is working.", calc["trend"]["working"])

    band = grading.absence_band(absences)
    band_rate = rates.get("absence_band", {}).get(band)
    if absences >= 11:
        add("high", "Your absences are high",
            f"You have missed {absences} classes. Plan to attend every lesson from now on and get the notes for the ones you missed. "
            + (f"In the data, {band_rate}% of students with 11 or more absences failed, against {rates.get('overall')}% overall." if band_rate is not None else ""),
            calc["absences"]["working"])
    elif absences >= 6:
        add("medium", "Watch your attendance",
            f"{absences} absences so far. Keep it from growing: arrange catch-up for the lessons you missed.", calc["absences"]["working"])
    elif absences == 0 and avg < grading.PASS_MARK:
        add("low", "Attendance is not the issue",
            "You attend every lesson but your marks are low, so the problem is how you study, not whether you come. Ask your teacher for feedback on your answers.",
            calc["absences"]["working"])

    if "next_grade" in calc:
        gap = grading.next_grade_up(avg)
        if gap and gap["gap"] <= 3 and avg >= grading.PASS_MARK:
            add("low", f"You are close to a {gap['letter']}", f"Only {gap['gap']:.1f} marks short. Small gains in the final can move you up a band.",
                calc["next_grade"]["working"])

    if "peer" in calc:
        pct = float(calc["peer"]["value"].split("than ")[1].split("%")[0])
        if pct < 25:
            add("medium", "You are in the lower quarter of your group",
                "Join a study group or tutoring for this course, and compare notes with a student who scores higher.",
                calc["peer"]["working"])

    if ml and ml.get("p_fail") is not None:
        p = ml["p_fail"]
        if p >= 50:
            add("high", "The model flags a high risk of failing", f"Estimated risk {p:.0f}%. Please act on the points above; "
                "this is an alert, not a verdict.", f"Model probability = {p:.1f}% (>= 50%)")
        elif p >= 25:
            add("medium", "The model sees a moderate risk of failing", f"Estimated risk {p:.0f}%. Tackle the biggest factor below first.",
                f"Model probability = {p:.1f}% (25-50%)")
        if ml.get("projected_final") is not None:
            add("low", "Projected final mark", f"The model projects a final mark of about {ml['projected_final']:.1f}% "
                f"({grading.letter_for(ml['projected_final'])}). It is a guide, typically within 5-10 marks.",
                f"{ml.get('projection_model')}: {ml['projected_final']:.1f}%")

    if "planner" in calc:
        add("low", "Your goal for the final", f"{calc['planner']['title']}: {calc['planner']['meaning']}", calc["planner"]["working"])

    if not any(s["priority"] in ("high", "medium") for s in out):
        add("good", "You are on track", "No warning signs in your record. Keep attending and revising.", "All indicators within normal range")
    out.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return out


def report_text(student: dict, indicators: dict, suggestions: list, ml: dict | None = None) -> str:
    """Plain-text version of the report (for copy / email)."""
    lines = [f"Academic progress report - {student.get('name')} ({student.get('course')})", ""]
    for c in indicators["calculations"]:
        lines += [f"{c['title']}: {c['value']}", f"  Formula: {c['formula']}", f"  Working: {c['working']}", ""]
    if ml and ml.get("p_fail") is not None:
        lines += [f"Estimated risk of failing: {ml['p_fail']:.0f}% ({ml.get('level')})", ""]
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
