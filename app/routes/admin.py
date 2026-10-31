"""
Admin pages: user accounts and the audit log.
"""

from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.audit import log_event, recent_events
from app.auth import ROLES, add_user, admin_required, current_user, delete_user, get_user, load_users, validate_password
from app.data.data_loader import get_student_by_id

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/users")
@admin_required
def users():
    df = load_users()
    return render_template("admin_users.html", users=df.drop(columns=["password_hash"]).to_dict("records"), roles=ROLES)


@admin_bp.route("/users/add", methods=["POST"])
@admin_required
def users_add():
    username = request.form.get("username", "").strip()
    role = request.form.get("role", "")
    password = request.form.get("password", "")
    student_id = request.form.get("student_id", "").strip()
    problem = None
    if not username.replace("_", "").replace(".", "").replace("-", "").isalnum() or not 3 <= len(username) <= 32:
        problem = "Username: 3-32 letters, numbers, . _ -"
    elif role not in ROLES:
        problem = "Choose a role."
    elif get_user(username) is not None:
        problem = "That username already exists."
    elif role == "student" and not (student_id.isdigit() and get_student_by_id(int(student_id))):
        problem = "A student account needs the ID of an existing student record."
    else:
        problem = validate_password(password)
    if problem:
        flash(problem, "danger")
    else:
        add_user(username, password, role=role, student_id=student_id if role == "student" else None)
        log_event("user_created", f"{username} ({role})")
        flash(f"Account '{username}' created.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<username>/delete", methods=["POST"])
@admin_required
def users_delete(username):
    if username == current_user()["username"]:
        flash("You cannot delete your own account.", "danger")
    elif delete_user(username):
        log_event("user_deleted", username)
        flash(f"Account '{username}' deleted.", "info")
    return redirect(url_for("admin.users"))


@admin_bp.route("/audit")
@admin_required
def audit():
    return render_template("admin_audit.html", events=recent_events(300))
