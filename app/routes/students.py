"""
Student routes: list, add, view, edit, delete and the printable report.

Access: staff (admin / teacher) manage every record; a student account sees only its own
record and report; visitors who are not logged in see none of this.
"""

import re

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app import grading
from app.audit import log_event
from app.auth import admin_required, can_view_student, current_user, login_required, staff_required
from app.data.analytics_calc import (
    build_class_stats, calculate_student_indicators, recommend_student, report_text,
)
from app.data.data_loader import add_student, delete_student, get_student_by_id, load_students, update_student
from app.ml.predictor import invalidate_cache, predict_student

students_bp = Blueprint("students", __name__, url_prefix="/students")

PAGE_SIZE = 25

COURSES = ("Mathematics", "Portuguese")
# Number fields: (low, high, whole number?)
NUMBER_FIELDS = {"t1_mark": (0, 100, False), "absences": (0, 100, True)}
OPTIONAL_NUMBER_FIELDS = {"t2_mark": (0, 100, False), "final_mark": (0, 100, False)}
CHOICE_FIELDS = {"course": COURSES}
NAME_PATTERN = re.compile(r"^[\w .'\-]{1,60}$", re.UNICODE)


def parse_student_form(form) -> tuple[dict, str | None]:
    """Validate a student form. Only known fields are read. Returns (data, error_message)."""
    data = {}

    name = form.get("name", "").strip()
    if name:
        name = name.lstrip("=+-@\t\r ")                  # never let a name start like a spreadsheet formula
        if not NAME_PATTERN.match(name):
            return {}, "Name may only contain letters, numbers, spaces and . ' - (60 characters at most)."
        data["name"] = name

    for key, allowed in CHOICE_FIELDS.items():
        value = form.get(key, "").strip()
        if value not in allowed:
            return {}, f"'{key}' must be one of: {', '.join(allowed)}."
        data[key] = value

    for key, (low, high, whole) in {**NUMBER_FIELDS, **OPTIONAL_NUMBER_FIELDS}.items():
        raw = str(form.get(key, "") or "").strip()
        if raw == "" and key in OPTIONAL_NUMBER_FIELDS:
            data[key] = None                                   # blank = not known yet
            continue
        try:
            number = float(raw)
        except ValueError:
            return {}, f"'{key}' must be a number."
        if number != number or number < low or number > high:          # NaN or out of range
            return {}, f"'{key}' must be between {low} and {high}."
        if whole:
            if number != int(number):
                return {}, f"'{key}' must be a whole number."
            number = int(number)
        else:
            number = round(number, 1)
        data[key] = number
    if data["t2_mark"] is None and data["final_mark"] is not None:
        return {}, "A final mark needs a term 2 mark first."
    return data, None


@students_bp.route("/")
@staff_required
def student_list():
    df = load_students()
    q = request.args.get("q", "").strip().lower()[:60]
    course = request.args.get("course", "")
    status = request.args.get("status", "")

    if not df.empty:
        if q:
            haystack = df["name"].astype(str).str.lower() + " " + df["student_id"].astype(str)
            df = df[haystack.str.contains(q, regex=False)]
        if course in COURSES:
            df = df[df["course"] == course]
        if status == "at_risk":
            df = df[df["at_risk"] == 1]
        elif status in ("Pass", "Fail", "In progress"):
            df = df[df["result"] == status]

    total = len(df)
    pages = max(1, -(-total // PAGE_SIZE))
    try:
        page = min(max(1, int(request.args.get("page", 1))), pages)
    except ValueError:
        page = 1
    chunk = df.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template(
        "students.html", students=chunk.to_dict("records") if total else [], total=total, page=page, pages=pages,
        q=request.args.get("q", "").strip()[:60], course=course, status=status,
        courses=COURSES,
    )


@students_bp.route("/add", methods=["POST"])
@staff_required
def student_add():
    data, error = parse_student_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_list"))
    new_id = add_student(data)
    invalidate_cache()
    log_event("student_added", f"id={new_id}")
    flash(f"Student added (ID {new_id}).", "success")
    return redirect(url_for("students.student_detail", student_id=new_id))


def _report_context(student, args):
    """Everything the report body needs: calculations, suggestions, model risk."""
    df = load_students()
    class_stats = build_class_stats(df) if not df.empty else {}
    try:
        target = float(args.get("target")) if args.get("target") else None
    except ValueError:
        target = None
    indicators = calculate_student_indicators(student, class_stats, target_average=target)
    ml = predict_student(student)
    ml = None if "error" in ml else ml
    suggestions = recommend_student(student, class_stats, indicators, ml)
    return {"student": student, "indicators": indicators, "suggestions": suggestions, "ml": ml,
            "text_report": report_text(student, indicators, suggestions, ml), "scale": grading.scale_table(),
            "pass_mark": grading.PASS_MARK}


def _student_or_404(student_id):
    if not can_view_student(student_id):
        abort(403)
    student = get_student_by_id(student_id)
    if student is None:
        abort(404)
    return student


@students_bp.route("/<int:student_id>")
@login_required
def student_detail(student_id):
    student = _student_or_404(student_id)
    log_event("student_viewed", f"id={student_id}")
    return render_template("student_detail.html", **_report_context(student, request.args),
                           courses=COURSES)


@students_bp.route("/<int:student_id>/report")
@login_required
def student_report(student_id):
    student = _student_or_404(student_id)
    log_event("report_viewed", f"id={student_id}")
    return render_template("report.html", **_report_context(student, request.args))


@students_bp.route("/my-report")
@login_required
def my_report():
    user = current_user()
    if user["role"] != "student" or not str(user.get("student_id") or "").isdigit():
        flash("This page is for student accounts linked to a student record.", "info")
        return redirect(url_for("main.dashboard"))
    return redirect(url_for("students.student_report", student_id=int(user["student_id"])))


@students_bp.route("/<int:student_id>/edit", methods=["POST"])
@staff_required
def student_edit(student_id):
    data, error = parse_student_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_detail", student_id=student_id))
    success = update_student(student_id, data)
    invalidate_cache()
    log_event("student_edited", f"id={student_id}")
    flash("Student updated." if success else "Student not found.", "success" if success else "danger")
    return redirect(url_for("students.student_detail", student_id=student_id))


@students_bp.route("/<int:student_id>/delete", methods=["POST"])
@admin_required
def student_delete(student_id):
    success = delete_student(student_id)
    invalidate_cache()
    log_event("student_deleted", f"id={student_id}")
    flash("Student deleted." if success else "Student not found.", "info" if success else "danger")
    return redirect(url_for("students.student_list"))
