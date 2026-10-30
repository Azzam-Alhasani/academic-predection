"""
Student management routes: list (search / filter / pages), add, view, edit, delete.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash

from app.auth import login_required
from app.data.analytics_calc import (
    build_class_stats,
    calculate_student_indicators,
    recommend_student,
)
from app.data.data_loader import (
    add_student,
    delete_student,
    get_student_by_id,
    load_students,
    update_student,
)
from app.ml.predictor import invalidate_cache

students_bp = Blueprint("students", __name__, url_prefix="/students")

PAGE_SIZE = 25

# Whole-number fields with their valid ranges. Only these fields are ever saved.
INT_FIELDS = {
    "age": (10, 30),
    "studytime": (1, 4),
    "failures": (0, 4),
    "absences": (0, 100),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}
CHOICE_FIELDS = {
    "subject": ("Math", "Portuguese"),
    "gender": ("Male", "Female"),
    "higher": ("yes", "no"),
    "internet": ("yes", "no"),
}


@students_bp.route("/")
def student_list():
    df = load_students()
    q = request.args.get("q", "").strip().lower()
    subject = request.args.get("subject", "")
    status = request.args.get("status", "")

    if not df.empty:
        if q:
            haystack = df["name"].astype(str).str.lower() + " " + df["student_id"].astype(str)
            df = df[haystack.str.contains(q, regex=False)]
        if subject in CHOICE_FIELDS["subject"]:
            df = df[df["subject"] == subject]
        if status == "at_risk":
            df = df[df["at_risk"] == 1]
        elif status == "pass":
            df = df[df["pass_fail"] == 1]
        elif status == "fail":
            df = df[df["pass_fail"] == 0]

    total = len(df)
    pages = max(1, -(-total // PAGE_SIZE))
    try:
        page = min(max(1, int(request.args.get("page", 1))), pages)
    except ValueError:
        page = 1
    chunk = df.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    students = chunk.to_dict("records") if total else []

    return render_template(
        "students.html",
        students=students, total=total, page=page, pages=pages,
        q=request.args.get("q", "").strip(), subject=subject, status=status,
    )


@students_bp.route("/add", methods=["POST"])
@login_required
def student_add():
    data, error = _parse_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_list"))
    new_id = add_student(data)
    invalidate_cache()
    flash(f"Student added (ID {new_id}).", "success")
    return redirect(url_for("students.student_detail", student_id=new_id))


@students_bp.route("/<int:student_id>")
def student_detail(student_id):
    student = get_student_by_id(student_id)
    if student is None:
        flash("Student not found.", "danger")
        return redirect(url_for("students.student_list"))

    df = load_students()
    class_stats = build_class_stats(df) if not df.empty else {}
    return render_template(
        "student_detail.html",
        student=student,
        indicators=calculate_student_indicators(student, class_stats),
        recommendations=recommend_student(student, class_stats),
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
    flash("Student updated successfully." if success else "Student not found.",
          "success" if success else "danger")
    return redirect(url_for("students.student_detail", student_id=student_id))


@students_bp.route("/<int:student_id>/delete", methods=["POST"])
@login_required
def student_delete(student_id):
    success = delete_student(student_id)
    invalidate_cache()
    flash("Student deleted." if success else "Student not found.", "info" if success else "danger")
    return redirect(url_for("students.student_list"))


def _parse_form(form) -> tuple[dict, str | None]:
    """
    Validate a student form. Only known fields are read (anything else is ignored).
    Returns (data, error_message).
    """
    data = {}

    name = form.get("name", "").strip()
    if len(name) > 60:
        return {}, "Name must be 60 characters or fewer."
    if name:
        data["name"] = name

    for key, allowed in CHOICE_FIELDS.items():
        value = form.get(key, "").strip()
        if value not in allowed:
            return {}, f"Field '{key}' must be one of: {', '.join(allowed)}."
        data[key] = value

    for key, (lo, hi) in INT_FIELDS.items():
        raw = form.get(key, "").strip()
        try:
            number = float(raw)
        except ValueError:
            return {}, f"Field '{key}' must be a number."
        if number != int(number) or number < lo or number > hi:
            return {}, f"Field '{key}' must be a whole number between {lo} and {hi}."
        data[key] = int(number)

    data["pass_fail"] = 1 if data["G3"] >= 10 else 0
    return data, None
