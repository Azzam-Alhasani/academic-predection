"""
Semester planner (public, stores nothing) and the model-selection results.

A visitor enters a major, their CGPA and credits so far, and the courses they are taking with their
attendance; the page returns expected marks, the withdraw and attendance what-ifs, and the calculations.
"""

from flask import Blueprint, render_template, request

from app import grading
from app.data import data_loader
from app.data.analytics_calc import plan_semester
from app.ml.preprocessing import SCENARIOS
from app.ml.predictor import get_training_size, load_selection

predictions_bp = Blueprint("predictions", __name__, url_prefix="/predictions")
ROWS = 8


def _catalogue():
    t = data_loader.load_tables()
    if t.empty:
        return [], []
    courses = t.courses.sort_values(["major", "course_code"])
    groups = []
    for major, g in courses.groupby("major", sort=False):
        groups.append({"major": major, "courses": g.to_dict("records")})
    groups.sort(key=lambda x: (x["major"] == "Common", x["major"]))
    return groups, sorted(t.students["major"].unique())


def _number(raw, low, high, label, whole=False):
    raw = (raw or "").strip()
    if raw == "":
        return None, None
    try:
        value = float(raw)
    except ValueError:
        return None, f"{label} must be a number."
    if not low <= value <= high:
        return None, f"{label} must be between {low} and {high}."
    return (int(value) if whole else value), None


@predictions_bp.route("/", methods=["GET", "POST"])
def predictions():
    study = load_selection()
    groups, majors = _catalogue()
    form = {"major": majors[0] if majors else "", "prior_cgpa": "", "prior_credits": "", "target": "",
            "rows": [{"course_code": "", "attendance": ""} for _ in range(ROWS)]}
    plan, error = None, None
    if request.method == "POST":
        f = request.form
        form.update(major=f.get("major", ""), prior_cgpa=f.get("prior_cgpa", ""), prior_credits=f.get("prior_credits", ""),
                    target=f.get("target", ""))
        form["rows"] = [{"course_code": f.get(f"course_{i}", ""), "attendance": f.get(f"attendance_{i}", "")} for i in range(ROWS)]
        valid = {c["course_code"] for g in groups for c in g["courses"]}
        cgpa, error = _number(f.get("prior_cgpa"), 0, 4, "CGPA")
        credits, e2 = _number(f.get("prior_credits"), 0, 300, "Credits so far")
        target, e3 = _number(f.get("target"), 0, 4, "Target CGPA")
        error = error or e2 or e3
        if not error and form["major"] not in majors:
            error = "Choose a major."
        if not error and (cgpa is None) != (not credits):
            error = "Enter both your CGPA and the credits attempted so far, or leave both empty if this is your first semester."
        chosen = []
        for r in form["rows"]:
            if not r["course_code"]:
                continue
            att, e = _number(r["attendance"], 0, 100, "Attendance")
            if e or att is None:
                error = error or e or "Enter the attendance % for every course you chose."
            elif r["course_code"] not in valid:
                error = error or "Choose courses from the list."
            else:
                chosen.append({"course_code": r["course_code"], "attendance": att, "attempt": 1})
        if not error and len({c["course_code"] for c in chosen}) != len(chosen):
            error = "Each course can be chosen once."
        if not error and len(chosen) < 1:
            error = "Choose at least one course."
        if not error:
            plan = plan_semester(form["major"], cgpa, credits or 0, chosen, target) or None
            if plan is None:
                error = "Not enough data to fit the models."
    return render_template("predictions.html", study=study, scenarios=SCENARIOS, form=form, groups=groups, majors=majors,
                           plan=plan, error=error, rows=range(ROWS), training_size=get_training_size())


@predictions_bp.route("/selection")
def model_selection():
    return render_template("model_selection.html", study=load_selection(), scenarios=SCENARIOS, scale=grading.scale_table())
