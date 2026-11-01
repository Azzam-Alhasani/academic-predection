"""
Semester planner: predictions, what-ifs, calculations with formulas, and suggestions.

Given a student's history (CGPA and credits so far) and the courses they are taking now, this works out

  * the expected mark, grade and chance of failing for every course (from the models),
  * how hard each course is compared with the student's other courses (difficulty measured from marks),
  * WITHDRAW what-if: if one course is withdrawn (W: no grade points, not attempted), how do the
    semester GPA and CGPA change, what is lost, and which course is the best one to drop,
  * ATTENDANCE what-if: what happens to the marks, the chance of failing and the GPA at 90% attendance,
  * a goal planner: the semester GPA needed to reach a target CGPA.

Every number comes with its formula and working so it can be checked by hand. The advice is a
statistical estimate learned from past students; it is not a promise and not a decision.
"""

import math

import numpy as np

from app import grading
from app.data import data_loader
from app.ml import predictor

TARGET_ATTENDANCE = 90.0
WITHDRAW_MIN_P_FAIL = 0.30          # only suggest dropping a course that is likely to be failed
WITHDRAW_MIN_GAIN = 0.03            # ... and only if the CGPA gain is worth the lost credits
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "good": 3}


def _num(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _pct(x) -> str:
    return f"{x * 100:.0f}%"


# ── building the plan ────────────────────────────────────────────────────────

def plan_semester(major: str, prior_cgpa, prior_attempted, courses: list, target_cgpa=None) -> dict:
    """
    courses: [{course_code, attendance, attempt}] for the semester in progress.
    Returns {} when there is nothing to plan or the models cannot be fitted.
    """
    tables = data_loader.load_tables()
    catalogue = tables.courses.set_index("course_code").to_dict("index") if not tables.courses.empty else {}
    courses = [c for c in courses if c.get("course_code") in catalogue]
    if not courses:
        return {}
    stats = data_loader.course_stats()
    stat = stats.set_index("course_code").to_dict("index") if not stats.empty else {}

    prior_attempted = _num(prior_attempted)
    prior_cgpa = None if prior_cgpa is None or (isinstance(prior_cgpa, float) and math.isnan(prior_cgpa)) else float(prior_cgpa)
    prior_quality = (prior_cgpa or 0.0) * prior_attempted
    total_credits = sum(catalogue[c["course_code"]]["credit_hours"] for c in courses)

    def record(c, attendance):
        info = catalogue[c["course_code"]]
        return {"prior_cgpa": prior_cgpa if prior_attempted > 0 else None, "prior_credits": prior_attempted,
                "credit_hours": info["credit_hours"], "load_credits": total_credits, "level": info["level"],
                "attempt": c.get("attempt", 1), "course_code": c["course_code"], "major": major, "attendance": attendance}

    now = predictor.predict_courses([record(c, _num(c.get("attendance"), 85.0)) for c in courses], "during")
    if now is None:
        return {}
    better = predictor.predict_courses(
        [record(c, max(_num(c.get("attendance"), 85.0), TARGET_ATTENDANCE)) for c in courses], "during")
    sd = predictor.regression_sd("during")

    rows = []
    for c, a, b in zip(courses, now, better):
        info, st = catalogue[c["course_code"]], stat.get(c["course_code"], {})
        att = _num(c.get("attendance"), 85.0)
        mark, p = a["mark"], a["p_fail"]
        mark2, p2 = max(mark, b["mark"]), min(p, b["p_fail"])           # attending more is never predicted to hurt
        rows.append({
            "course_code": c["course_code"], "course_name": info["course_name"], "credits": info["credit_hours"],
            "attendance": att, "attempt": c.get("attempt", 1),
            "tier": st.get("tier", "Medium"), "difficulty": st.get("difficulty"), "class_mean": st.get("mean_mark"),
            "class_fail": st.get("fail_rate"),
            "mark": round(mark, 1), "grade": grading.letter_for(mark), "e_points": grading.expected_points(mark, sd),
            "p_fail": p, "mark_att": round(mark2, 1), "e_points_att": grading.expected_points(mark2, sd), "p_fail_att": p2,
        })

    def totals(items, key="e_points"):
        credits = sum(r["credits"] for r in items)
        quality = sum(r["credits"] * r[key] for r in items)
        sgpa = quality / credits if credits else None
        cgpa = (prior_quality + quality) / (prior_attempted + credits) if prior_attempted + credits > 0 else None
        return credits, quality, sgpa, cgpa

    credits, quality, sgpa, cgpa = totals(rows)
    _, quality_att, sgpa_att, cgpa_att = totals(rows, "e_points_att")
    p_any = 1 - float(np.prod([1 - r["p_fail"] for r in rows]))

    withdraw = []
    if len(rows) >= 2:
        for r in rows:
            rest = [x for x in rows if x is not r]
            c2, _, s2, g2 = totals(rest)
            withdraw.append({"course_code": r["course_code"], "course_name": r["course_name"], "p_fail": r["p_fail"],
                             "credits_left": c2, "sgpa": s2, "cgpa": g2, "d_sgpa": s2 - sgpa,
                             "d_cgpa": (g2 - cgpa) if g2 is not None and cgpa is not None else 0.0,
                             "load_after": grading.load_status(c2), "full_time": c2 >= grading.LIGHT_LOAD})
        withdraw.sort(key=lambda w: -w["d_cgpa"])

    target = _default_target(prior_cgpa) if target_cgpa is None else max(0.0, min(4.0, float(target_cgpa)))
    needed, reachable = (grading.required_sgpa(prior_cgpa or 0.0, prior_attempted, target, credits) if prior_attempted > 0 else (None, True))

    plan = {"courses": rows, "credits": credits, "load": grading.load_status(credits), "sgpa": sgpa, "cgpa": cgpa,
            "sgpa_att": sgpa_att, "cgpa_att": cgpa_att, "p_any_fail": p_any, "prior_cgpa": prior_cgpa,
            "prior_attempted": prior_attempted, "prior_quality": prior_quality, "withdraw": withdraw, "sd": sd,
            "target": target, "needed_sgpa": needed, "reachable": reachable,
            "models": predictor.model_labels("during"),
            "hard_courses": [r for r in rows if r["tier"] == "Hard"]}
    plan["calculations"] = _calculations(plan)
    plan["suggestions"] = _suggestions(plan)
    return plan


def _default_target(prior_cgpa):
    if prior_cgpa is None:
        return 3.0
    return 2.0 if prior_cgpa < 1.9 else round(min(4.0, prior_cgpa + 0.2), 1)


# ── calculations (formula + working) ─────────────────────────────────────────

def _calculations(p) -> list:
    calc = []
    rows = p["courses"]

    def add(key, title, value, formula, working, meaning):
        calc.append({"key": key, "title": title, "value": value, "formula": formula, "working": working, "meaning": meaning})

    add("load", "Credit load this semester", f"{p['credits']:.0f} credits - {p['load']}",
        "Credits = sum of the credit hours of the courses taken (a withdrawn course is not counted)",
        " + ".join(f"{r['credits']}" for r in rows) + f" = {p['credits']:.0f}",
        f"Normal load is {grading.LIGHT_LOAD}-{grading.NORMAL_MAX} credits (assumed limits).")
    terms = " + ".join(f"{r['credits']} x {r['e_points']:.2f}" for r in rows)
    add("sgpa", "Expected semester GPA (SGPA)", f"{p['sgpa']:.2f} / 4.00",
        "SGPA = sum(credit hours x expected grade points) / credits attempted",
        f"({terms}) / {p['credits']:.0f} = {p['sgpa']:.2f}",
        f"Expected grade points are averaged over the model's likely marks (typical error {p['sd']:.1f} marks), not taken from one guess.")
    if p["cgpa"] is not None:
        if p["prior_attempted"] > 0:
            add("cgpa", "Expected CGPA after this semester", f"{p['cgpa']:.2f} / 4.00 (now {p['prior_cgpa']:.2f})",
                "CGPA = (quality points so far + this semester's) / (credits attempted so far + this semester's)",
                f"({p['prior_quality']:.1f} + {p['sgpa'] * p['credits']:.1f}) / ({p['prior_attempted']:.0f} + {p['credits']:.0f}) = {p['cgpa']:.2f}",
                f"Standing would be: {grading.standing(p['cgpa'])} (good standing from {grading.GOOD_STANDING_CGPA}).")
        else:
            add("cgpa", "CGPA after this semester", f"{p['cgpa']:.2f} / 4.00", "CGPA = SGPA in the first semester",
                f"{p['sgpa']:.2f}", "There is no earlier record, so this is the first CGPA.")
    p_terms = " x ".join(f"(1 - {r['p_fail']:.2f})" for r in rows)
    add("risk", "Chance of failing at least one course", _pct(p["p_any_fail"]),
        "P(at least one F) = 1 - product of (1 - P(fail) of each course)",
        f"1 - {p_terms} = {p['p_any_fail']:.2f}", "A failed course (below C) earns 0 points and must be repeated.")

    hard = sorted(rows, key=lambda r: -(r["difficulty"] if r["difficulty"] is not None else -99))
    d = hard[0]
    if d["difficulty"] is not None:
        add("difficulty", "Hardest course in your schedule", f"{d['course_name']} (difficulty {d['difficulty']:+.1f})",
            "Difficulty = average over past students of (their mean mark in other courses - their mark here)",
            "; ".join(f"{r['course_name']} {r['difficulty']:+.1f}" for r in hard if r["difficulty"] is not None),
            "Positive = students score that many marks lower here than in their other courses (measured from the data, not set by hand).")

    if p["withdraw"]:
        w = p["withdraw"][0]
        add("withdraw", f"Withdraw what-if: {w['course_name']}", f"CGPA {p['cgpa']:+.2f} -> {w['cgpa']:.2f} ({w['d_cgpa']:+.2f})",
            "SGPA without the course = sum over the other courses(credits x points) / their credits; a W has no points and is not attempted",
            f"SGPA {p['sgpa']:.2f} -> {w['sgpa']:.2f} ; credits {p['credits']:.0f} -> {w['credits_left']:.0f} ({w['load_after']})",
            "Best option shown; the full table lists every course.")
    gain = (p["sgpa_att"] or 0) - (p["sgpa"] or 0)
    low = [r for r in rows if r["attendance"] < TARGET_ATTENDANCE]
    if low:
        add("attendance", f"Attendance what-if: {TARGET_ATTENDANCE:.0f}% in every course", f"SGPA {p['sgpa']:.2f} -> {p['sgpa_att']:.2f} ({gain:+.2f})",
            "Re-run the model with attendance raised to the target; SGPA gain = sum(credits x change in expected points) / credits",
            "; ".join(f"{r['course_name']}: {r['attendance']:.0f}% -> mark {r['mark']:.0f} -> {r['mark_att']:.0f}" for r in low),
            "A statistical association learned from past students; attending more is assumed never to lower a mark.")
    if p["needed_sgpa"] is not None:
        meaning = ("Reachable this semester." if p["reachable"] else "Not reachable in one semester (it would need more than 4.00).")
        add("planner", f"Goal planner: CGPA {p['target']:.2f}", f"SGPA of {p['needed_sgpa']:.2f} needed (expected {p['sgpa']:.2f})",
            "Needed SGPA = (target x (credits so far + n) - CGPA x credits so far) / n",
            f"({p['target']:.2f} x ({p['prior_attempted']:.0f} + {p['credits']:.0f}) - {p['prior_cgpa']:.2f} x {p['prior_attempted']:.0f}) / {p['credits']:.0f} = {p['needed_sgpa']:.2f}",
            meaning)
    return calc


# ── suggestions ──────────────────────────────────────────────────────────────

def _suggestions(p) -> list:
    out = []
    rows = p["courses"]

    def add(priority, title, text, why):
        out.append({"priority": priority, "title": title, "text": text, "why": why})

    risky = [w for w in p["withdraw"] if w["p_fail"] >= WITHDRAW_MIN_P_FAIL and w["d_cgpa"] >= WITHDRAW_MIN_GAIN]
    by_code = {r["course_code"]: r for r in rows}
    if risky:
        w = risky[0]
        r = by_code[w["course_code"]]
        others = [x for x in risky[1:]]
        fixable = r["attendance"] < 85 and r["p_fail_att"] < 0.25
        compare = ""
        if others:
            o = others[0]
            ro = by_code[o["course_code"]]
            compare = (f" {ro['course_name']} is also risky (predicted {ro['mark']:.0f}%, {_pct(o['p_fail'])} chance of failing), "
                       f"but dropping {r['course_name']} helps your CGPA more ({w['d_cgpa']:+.2f} against {o['d_cgpa']:+.2f}). ")
        if fixable:
            add("high", f"Fix attendance in {r['course_name']} before withdrawing",
                f"{r['course_name']} is your riskiest course ({_pct(w['p_fail'])} chance of failing at {r['attendance']:.0f}% attendance). "
                f"At {TARGET_ATTENDANCE:.0f}% attendance the model puts the chance at {_pct(r['p_fail_att'])} (predicted mark {r['mark']:.0f}% to {r['mark_att']:.0f}%).{compare}"
                "If you cannot attend regularly, withdrawing is the fallback.",
                f"P(fail) {w['p_fail']:.2f} -> {r['p_fail_att']:.2f} with attendance {r['attendance']:.0f}% -> {TARGET_ATTENDANCE:.0f}%")
        else:
            cost = (f" You would be on {w['credits_left']:.0f} credits ({w['load_after']})"
                    + ("; that is below the assumed full-time minimum, so check with your advisor first." if not w["full_time"] else "."))
            add("high", f"Consider withdrawing {r['course_name']}",
                f"{r['course_name']} is predicted at {r['mark']:.0f}% ({r['grade']}) with a {_pct(w['p_fail'])} chance of failing. A failed course adds 0 points for "
                f"{r['credits']} credits, while a withdrawal (W) is not counted in your GPA. Withdrawing lifts your expected CGPA from {p['cgpa']:.2f} to {w['cgpa']:.2f} ({w['d_cgpa']:+.2f}).{compare or ' '}"
                f"The cost: you earn {r['credits']} fewer credits now and must take the course again later.{cost} Check the official withdrawal deadline.",
                f"CGPA {p['cgpa']:.2f} -> {w['cgpa']:.2f}; P(fail) = {w['p_fail']:.2f}")
    elif len(rows) >= 2:
        worst = max(rows, key=lambda r: r["p_fail"])
        add("good", "No course needs to be withdrawn",
            f"Every course is expected to pass. The riskiest is {worst['course_name']} ({_pct(worst['p_fail'])} chance of failing), "
            "and withdrawing a course you are likely to pass would only cost credits.", f"max P(fail) = {worst['p_fail']:.2f}")

    shown = {risky[0]["course_code"]} if risky else set()
    gains = []
    for r in rows:
        if r["attendance"] < 85 and r["course_code"] not in shown:
            d_gpa = r["credits"] * (r["e_points_att"] - r["e_points"]) / p["credits"]
            if d_gpa >= 0.03 or (r["p_fail"] - r["p_fail_att"]) >= 0.08:
                gains.append((d_gpa, r))
    for d_gpa, r in sorted(gains, key=lambda t: -t[0])[:3]:
        add("medium", f"Attend more in {r['course_name']}",
            f"Your attendance is {r['attendance']:.0f}%. At {TARGET_ATTENDANCE:.0f}% the predicted mark rises from {r['mark']:.0f}% to {r['mark_att']:.0f}%, "
            f"the chance of failing falls from {_pct(r['p_fail'])} to {_pct(r['p_fail_att'])}, and your semester GPA gains about {d_gpa:+.2f}.",
            f"expected points {r['e_points']:.2f} -> {r['e_points_att']:.2f} x {r['credits']} credits / {p['credits']:.0f}")

    hard = p["hard_courses"]
    if len(hard) >= 3:
        names = ", ".join(r["course_name"] for r in hard)
        add("medium", "Three or more of your courses are among the hardest",
            f"{names}. Hard courses are where past students lost the most marks compared with their other courses. "
            "If you can, move one of them to a lighter semester instead of taking them together.",
            f"{len(hard)} courses with difficulty tier Hard")
    if p["load"] == "Overload":
        add("medium", "You are taking an overload",
            f"{p['credits']:.0f} credits is above the normal {grading.NORMAL_MAX}. A heavy load leaves less time per course; "
            "consider deferring the easiest-to-postpone course.", f"credits = {p['credits']:.0f} > {grading.NORMAL_MAX}")
    if p["prior_cgpa"] is not None and p["prior_cgpa"] < grading.GOOD_STANDING_CGPA:
        level = "high" if p["prior_cgpa"] < grading.WARNING_CGPA else "medium"
        add(level, "Your CGPA is below good standing",
            f"CGPA {p['prior_cgpa']:.2f} is under {grading.GOOD_STANDING_CGPA}. To reach {p['target']:.2f} you need an SGPA of {p['needed_sgpa']:.2f} this semester"
            + (" - not reachable in one semester, so plan two semesters." if not p["reachable"] else ".") + " Meet your academic advisor.",
            f"needed SGPA = {p['needed_sgpa']:.2f}")
    if p["p_any_fail"] >= 0.5:
        add("medium", "You are likely to fail at least one course",
            f"The chance of at least one F this semester is {_pct(p['p_any_fail'])}. Use the withdraw and attendance what-ifs above.",
            f"P(at least one F) = {p['p_any_fail']:.2f}")
    if not any(s["priority"] in ("high", "medium") for s in out):
        add("good", "You are on track", f"Expected SGPA {p['sgpa']:.2f}; no warning signs in this schedule.", "All indicators within normal range")
    out.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return out


# ── stored students ──────────────────────────────────────────────────────────

def plan_for_student(student_id: int, target_cgpa=None) -> dict:
    """Plan for the semester in progress of a stored student; {} when there is none."""
    t = data_loader.load_tables()
    student = data_loader.get_student(student_id)
    if student is None:
        return {}
    cur = t.enrollments[(t.enrollments["student_id"] == int(student_id)) & (t.enrollments["status"] == "In progress")]
    courses = [{"course_code": r.course_code, "attendance": r.attendance, "attempt": int(r.attempt)} for r in cur.itertuples()]
    plan = plan_semester(student["major"], student.get("cgpa"), student.get("attempted"), courses, target_cgpa)
    if plan:
        plan["retakes"] = retakes_needed(student_id)
        if plan["retakes"]:
            names = ", ".join(r["course_name"] for r in plan["retakes"])
            plan["suggestions"].append({"priority": "medium", "title": "Courses you still need to retake",
                                        "text": f"{names}: the latest attempt was not passed and it is not in this semester's schedule. Plan them for a coming semester.",
                                        "why": "latest attempt result = F or W, no later pass"})
            plan["suggestions"].sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
    return plan


def retakes_needed(student_id: int) -> list:
    t = data_loader.load_tables()
    e = t.enrollments[t.enrollments["student_id"] == int(student_id)]
    passed = set(e.loc[e["result"] == "P", "course_code"])
    current = set(e.loc[e["status"] == "In progress", "course_code"])
    last = e[e["is_latest"] & e["result"].isin(["F", "W"])]
    return [{"course_code": r.course_code, "course_name": r.course_name, "result": r.result}
            for r in last.itertuples() if r.course_code not in passed and r.course_code not in current]


def report_text(student: dict, plan: dict) -> str:
    """Plain-text version of the report (for copy / email)."""
    lines = [f"Academic progress report - {student.get('name')} ({student.get('major')})", ""]
    if student.get("cgpa") is not None:
        lines += [f"CGPA {student['cgpa']:.2f} - {student.get('standing')} - {student.get('attempted', 0):.0f} credits attempted, "
                  f"{student.get('earned', 0):.0f} earned", ""]
    if plan:
        for c in plan["calculations"]:
            lines += [f"{c['title']}: {c['value']}", f"  Formula: {c['formula']}", f"  Working: {c['working']}", ""]
        lines.append("Suggestions")
        for i, s in enumerate(plan["suggestions"], 1):
            lines.append(f"{i}. [{s['priority'].upper()}] {s['title']}: {s['text']}")
    return "\n".join(lines)
