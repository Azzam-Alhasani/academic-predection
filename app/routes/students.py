"""
Student management routes: list, add, view, edit, delete.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.data.data_loader import (
    load_students,
    get_student_by_id,
    add_student,
    update_student,
    delete_student,
)
from app.ml.predictor import invalidate_cache
from app.auth import login_required
from app.data.analytics_calc import calculate_student_indicators, recommend_student, build_class_stats

students_bp = Blueprint("students", __name__, url_prefix="/students")

SUBJECT_COLS = ["math", "science", "english", "history", "cs"]

# Fields that must be numeric, with their valid ranges (min, max).
NUMERIC_FIELDS = {
    "age": (10, 30),
    "attendance_pct": (0, 100),
    "assignments_submitted": (0, 12),
    "study_hours_per_week": (0, 20),
    "previous_gpa": (0, 4),
    "previous_term_avg": (0, 100),
    "math": (0, 100),
    "science": (0, 100),
    "english": (0, 100),
    "history": (0, 100),
    "cs": (0, 100),
}

# Fields kept in the CSV exactly (subject scores are numeric and validated above).
NON_NUMERIC_FIELDS = {"name", "gender"}


@students_bp.route("/")
def student_list():
    df = load_students()
    students = df.to_dict("records") if not df.empty else []
    return render_template("students.html", students=students)


@students_bp.route("/add", methods=["POST"])
@login_required
def student_add():
    data, error = _parse_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_list"))
    add_student(data)
    invalidate_cache()
    flash("Student added successfully.", "success")
    return redirect(url_for("students.student_list"))


@students_bp.route("/<int:student_id>")
def student_detail(student_id):
    student = get_student_by_id(student_id)
    if student is None:
        flash("Student not found.", "danger")
        return redirect(url_for("students.student_list"))

    df = load_students()
    class_stats = build_class_stats(df) if not df.empty else {}
    indicators = calculate_student_indicators(student, class_stats)
    recommendations = recommend_student(student, class_stats)
    return render_template(
        "student_detail.html",
        student=student,
        indicators=indicators,
        recommendations=recommendations,
    )


@students_bp.route("/<int:student_id>/edit", methods=["POST"])
@login_required
def student_edit(student_id):
    data, error = _parse_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_detail", student_id=student_id))
    success = update_student(student_id, data)
    invalidate_cache()
    if success:
        flash("Student updated successfully.", "success")
    else:
        flash("Student not found.", "danger")
    return redirect(url_for("students.student_detail", student_id=student_id))


@students_bp.route("/<int:student_id>/delete", methods=["POST"])
@login_required
def student_delete(student_id):
    success = delete_student(student_id)
    invalidate_cache()
    if success:
        flash("Student deleted.", "info")
    else:
        flash("Student not found.", "danger")
    return redirect(url_for("students.student_list"))


def _parse_form(form) -> tuple[dict, str | None]:
    """
    Parse and validate a student form. Returns (data, error_message).
    Numeric fields are clamped/validated to their allowed ranges, and the name
    is required. A validation error returns (None, message).
    """
    data = {}
    for key in form:
        # Skip the CSRF hidden field and other non-student fields.
        if key == "csrf_token":
            continue
        val = form[key].strip()

        if key in NUMERIC_FIELDS:
            lo, hi = NUMERIC_FIELDS[key]
            try:
                number = float(val)
            except ValueError:
                return {}, f"Field '{key}' must be a number."
            if number < lo or number > hi:
                return {}, f"Field '{key}' must be between {lo} and {hi} (got {number:.1f})."
            # Keep integers for whole-number fields, floats otherwise.
            if key in ["age", "assignments_submitted"]:
                data[key] = int(number)
            else:
                data[key] = number
        else:
            data[key] = val

    if not data.get("name"):
        return {}, "Student name is required."

    # Derive pass_fail from subject scores.
    scores = [data.get(s, 0) for s in SUBJECT_COLS if s in data]
    if scores:
        avg = sum(scores) / len(scores)
        data["pass_fail"] = 1 if avg >= 50 else 0

    return data, None