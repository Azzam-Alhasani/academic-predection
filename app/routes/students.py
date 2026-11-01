"""
Student routes: list, add, transcript + semester plan, edit, delete, and enrollment records.

Access: staff (admin / teacher) manage every record; a student account sees only its own transcript and
plan; visitors who are not logged in see none of this.
"""

import re

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app import grading
from app.audit import log_event
from app.auth import admin_required, can_view_student, current_user, login_required, staff_required
from app.data import data_loader
from app.data.analytics_calc import plan_for_student, report_text
from app.ml.predictor import invalidate_cache

students_bp = Blueprint("students", __name__, url_prefix="/students")

PAGE_SIZE = 25
GENDERS = ("Male", "Female")
SEMESTER_PATTERN = re.compile(r"^(20\d\d) (Spring|Summer|Fall)$")
NAME_PATTERN = re.compile(r"^[\w .'\-]{1,60}$", re.UNICODE)


def majors() -> list:
    t = data_loader.load_tables()
    return sorted(t.students["major"].unique()) if not t.empty else []


def parse_student_form(form) -> tuple[dict, str | None]:
    """Validate a student form; only known fields are read."""
    data = {}
    name = form.get("name", "").strip().lstrip("=+-@\t\r ")          # never start like a spreadsheet formula
    if name:
        if not NAME_PATTERN.match(name):
            return {}, "Name may only contain letters, numbers, spaces and . ' - (60 characters at most)."
        data["name"] = name
    if form.get("gender") not in GENDERS:
        return {}, f"Gender must be one of: {', '.join(GENDERS)}."
    data["gender"] = form["gender"]
    if form.get("major") not in majors():
        return {}, "Choose one of the listed majors."
    data["major"] = form["major"]
    entry = form.get("entry_semester", "").strip()
    if not SEMESTER_PATTERN.match(entry):
        return {}, "Entry semester must look like '2025 Fall' (Spring, Summer or Fall)."
    data["entry_semester"] = entry
    return data, None


def parse_enrollment_form(form, require_course=True) -> tuple[dict, str | None]:
    """Validate an enrollment (one course attempt)."""
    data = {}
    if require_course:
        t = data_loader.load_tables()
        if form.get("course_code") not in set(t.courses["course_code"]):
            return {}, "Choose a course from the list."
        data["course_code"] = form["course_code"]
        semester = form.get("semester", "").strip()
        if not SEMESTER_PATTERN.match(semester):
            return {}, "Semester must look like '2026 Fall' (Spring, Summer or Fall)."
        data["semester"] = semester
    status = form.get("status", "")
    if status not in data_loader.STATUSES:
        return {}, f"Status must be one of: {', '.join(data_loader.STATUSES)}."
    data["status"] = status
    try:
        attendance = float(form.get("attendance", ""))
    except ValueError:
        return {}, "Attendance must be a number between 0 and 100."
    if not 0 <= attendance <= 100:
        return {}, "Attendance must be between 0 and 100."
    data["attendance"] = round(attendance, 1)
    raw = form.get("mark", "").strip()
    if status == "Completed":
        try:
            mark = float(raw)
        except ValueError:
            return {}, "A completed course needs a mark between 0 and 100."
        if not 0 <= mark <= 100:
            return {}, "The mark must be between 0 and 100."
        data["mark"] = int(round(mark))
    else:
        data["mark"] = None
    return data, None


@students_bp.route("/")
@staff_required
def student_list():
    t = data_loader.load_tables()
    df = t.summary
    q = request.args.get("q", "").strip().lower()[:60]
    major = request.args.get("major", "")
    status = request.args.get("status", "")
    if not df.empty:
        if q:
            haystack = df["name"].astype(str).str.lower() + " " + df["student_id"].astype(str)
            df = df[haystack.str.contains(q, regex=False)]
        if major in majors():
            df = df[df["major"] == major]
        if status == "at_risk":
            df = df[df["at_risk"] == 1]
        elif status in ("Good standing", "Academic warning", "Academic probation", "No courses yet"):
            df = df[df["standing"] == status]
    total = len(df)
    pages = max(1, -(-total // PAGE_SIZE))
    try:
        page = min(max(1, int(request.args.get("page", 1))), pages)
    except ValueError:
        page = 1
    chunk = df.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template("students.html", students=chunk.to_dict("records") if total else [], total=total, page=page,
                           pages=pages, q=request.args.get("q", "").strip()[:60], major=major, status=status,
                           majors=majors(), genders=GENDERS, current_semester=t.current_semester or "2026 Fall")


@students_bp.route("/add", methods=["POST"])
@staff_required
def student_add():
    data, error = parse_student_form(request.form)
    if error:
        flash(error, "danger")
        return redirect(url_for("students.student_list"))
    new_id = data_loader.add_student(data)
    invalidate_cache()
    log_event("student_added", f"id={new_id}")
    flash(f"Student added (ID {new_id}). Add their courses below.", "success")
    return redirect(url_for("students.student_detail", student_id=new_id))


def _student_or_404(student_id):
    if not can_view_student(student_id):
        abort(403)
    student = data_loader.get_student(student_id)
    if student is None:
        abort(404)
    return student


def _context(student_id, args):
    student = _student_or_404(student_id)
    try:
        target = float(args.get("target")) if args.get("target") else None
    except ValueError:
        target = None
    plan = plan_for_student(student_id, target)
    t = data_loader.load_tables()
    return {"student": student, "plan": plan, "transcript": data_loader.transcript(student_id),
            "text_report": report_text(student, plan), "scale": grading.scale_table(), "pass_points": grading.PASS_POINTS,
            "majors": majors(), "genders": GENDERS, "courses": t.courses.sort_values(["major", "course_code"]).to_dict("records"),
            "statuses": data_loader.STATUSES, "current_semester": t.current_semester or "2026 Fall"}


@students_bp.route("/<int:student_id>")
@login_required
def student_detail(student_id):
    context = _context(student_id, request.args)
    log_event("student_viewed", f"id={student_id}")
    return render_template("student_detail.html", **context)


@students_bp.route("/<int:student_id>/report")
@login_required
def student_report(student_id):
    context = _context(student_id, request.args)
    log_event("report_viewed", f"id={student_id}")
    return render_template("report.html", **context)


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
    ok = data_loader.update_student(student_id, data)
    invalidate_cache()
    log_event("student_edited", f"id={student_id}")
    flash("Student updated." if ok else "Student not found.", "success" if ok else "danger")
    return redirect(url_for("students.student_detail", student_id=student_id))


@students_bp.route("/<int:student_id>/delete", methods=["POST"])
@admin_required
def student_delete(student_id):
    ok = data_loader.delete_student(student_id)
    invalidate_cache()
    log_event("student_deleted", f"id={student_id}")
    flash("Student and their course records deleted." if ok else "Student not found.", "info" if ok else "danger")
    return redirect(url_for("students.student_list"))


@students_bp.route("/<int:student_id>/enrollments/add", methods=["POST"])
@staff_required
def enrollment_add(student_id):
    if data_loader.get_student(student_id) is None:
        abort(404)
    data, error = parse_enrollment_form(request.form)
    if error:
        flash(error, "danger")
    else:
        data_loader.add_enrollment({**data, "student_id": student_id})
        invalidate_cache()
        log_event("enrollment_added", f"student={student_id} course={data['course_code']}")
        flash("Course record added.", "success")
    return redirect(url_for("students.student_detail", student_id=student_id))


def _owner_of(enrollment_id):
    enr = data_loader.load_tables().enrollments
    row = enr[enr["enrollment_id"] == enrollment_id]
    if row.empty:
        abort(404)
    return int(row.iloc[0]["student_id"])


@students_bp.route("/enrollments/<int:enrollment_id>/edit", methods=["POST"])
@staff_required
def enrollment_edit(enrollment_id):
    student_id = _owner_of(enrollment_id)
    data, error = parse_enrollment_form(request.form, require_course=False)
    if error:
        flash(error, "danger")
    else:
        data_loader.update_enrollment(enrollment_id, data)
        invalidate_cache()
        log_event("enrollment_edited", f"id={enrollment_id}")
        flash("Course record updated.", "success")
    return redirect(url_for("students.student_detail", student_id=student_id))


@students_bp.route("/enrollments/<int:enrollment_id>/delete", methods=["POST"])
@staff_required
def enrollment_delete(enrollment_id):
    student_id = _owner_of(enrollment_id)
    data_loader.delete_enrollment(enrollment_id)
    invalidate_cache()
    log_event("enrollment_deleted", f"id={enrollment_id}")
    flash("Course record deleted.", "info")
    return redirect(url_for("students.student_detail", student_id=student_id))
