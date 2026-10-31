"""
Authentication and role-based access control.

Roles
  admin    - everything, including user accounts and the audit log
  teacher  - view and edit all student records, recommendations, reports
  student  - sees only their own record and report (account linked to a student_id)
Anyone not logged in sees aggregate pages only (dashboard, analytics, model information).

Accounts live in app/data/users.csv with salted, hashed passwords (never plain text).
"""

import os
import re
import threading
import time
from functools import wraps

import pandas as pd
from flask import (
    Blueprint, abort, current_app, flash, redirect, render_template, request, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from app.audit import log_event

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

ROLES = ("admin", "teacher", "student")
STAFF_ROLES = ("admin", "teacher")
USER_COLUMNS = ["username", "password_hash", "role", "student_id"]
DEV_ADMIN_USERNAME = "admin"
DEV_ADMIN_PASSWORD = "admin123"
COMMON_PASSWORDS = {"password", "password1", "12345678", "123456789", "qwertyuiop", "admin123", "letmein123",
                    "iloveyou1", "welcome123", "administrator"}

COMMON_WORDS = ("password", "qwerty", "letmein", "welcome", "admin", "iloveyou", "123456", "abc123")

# Failed login attempts: {"ip:<addr>" or "user:<name>": [timestamps]}. In memory is enough for a
# single-process deployment and keeps brute-force guessing slow.
_failed_logins: dict = {}
_users_lock = threading.RLock()
# Hash of a random string: checked when the username does not exist, so a wrong username and a
# wrong password take the same time (no account enumeration by timing).
_DUMMY_HASH = generate_password_hash("not-a-real-password")


# ── password rules ───────────────────────────────────────────────────────────

def validate_password(password: str, production: bool | None = None) -> str | None:
    """Return a message describing what is wrong, or None when the password is acceptable."""
    if production is None:
        production = not current_app.config.get("DEBUG")
    minimum = 10 if production else 8
    if len(password) < minimum:
        return f"Use at least {minimum} characters."
    if (password.lower() in COMMON_PASSWORDS or len(set(password)) < 4
            or any(word in password.lower() for word in COMMON_WORDS)):
        return "That password is too easy to guess."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Use both letters and numbers."
    return None


# ── user store ───────────────────────────────────────────────────────────────

def _users_file():
    # __file__ (not current_app.root_path) so create_users.py also works without a Flask app.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.csv")


def load_users() -> pd.DataFrame:
    path = _users_file()
    if not os.path.exists(path):
        return pd.DataFrame(columns=USER_COLUMNS)
    try:
        df = pd.read_csv(path, dtype={"student_id": "object"})
    except Exception:
        return pd.DataFrame(columns=USER_COLUMNS)
    for col in USER_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["role"] = df["role"].fillna("admin")
    df["student_id"] = df["student_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return df[USER_COLUMNS]


def _save_users(df: pd.DataFrame) -> None:
    df[USER_COLUMNS].to_csv(_users_file(), index=False)


def has_accounts() -> bool:
    return not load_users().empty


def get_user(username: str) -> dict | None:
    df = load_users()
    if df.empty:
        return None
    rows = df[df["username"] == username]
    return None if rows.empty else rows.iloc[0].to_dict()


def add_user(username: str, password: str, role: str = "admin", student_id: str | int | None = None) -> None:
    """Create or replace an account (password is stored hashed)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    with _users_lock:
        df = load_users()
        record = {"username": username, "password_hash": generate_password_hash(password), "role": role,
                  "student_id": "" if student_id in (None, "") else str(student_id)}
        if not df.empty and (df["username"] == username).any():
            for key, value in record.items():
                df.loc[df["username"] == username, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        _save_users(df)


def delete_user(username: str) -> bool:
    with _users_lock:
        df = load_users()
        remaining = df[df["username"] != username]
        if len(remaining) == len(df):
            return False
        _save_users(remaining)
        return True


def verify_user(username: str, password: str) -> bool:
    user = get_user(username)
    if user is None:
        check_password_hash(_DUMMY_HASH, password)       # equalise timing
        return False
    return check_password_hash(user["password_hash"], password)


def ensure_dev_admin(app) -> None:
    """Development only: if no account exists, create admin / admin123 so a fresh install can log in."""
    if app.config.get("DEBUG") and not has_accounts():
        add_user(DEV_ADMIN_USERNAME, DEV_ADMIN_PASSWORD, role="admin")
        print(f" * First run: created the development login  {DEV_ADMIN_USERNAME} / {DEV_ADMIN_PASSWORD}")


def ensure_admin_from_env(app=None) -> None:
    """
    Create the admin account from ADMIN_PASSWORD (hosts without a shell, e.g. Render) when it does
    not exist yet. The password must satisfy the password rules; a weak one is refused.
    """
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        return
    username = os.environ.get("ADMIN_USERNAME", "admin")
    df = load_users()
    if not df.empty and (df["username"] == username).any():
        return
    production = not (app.config.get("DEBUG") if app is not None else False)
    problem = validate_password(password, production=production)
    if problem:
        print(f" ! ADMIN_PASSWORD was not used: {problem}")
        return
    add_user(username, password, role="admin")


# ── session helpers and decorators ───────────────────────────────────────────

def current_user() -> dict | None:
    username = session.get("username")
    return get_user(username) if username else None


def is_staff(user: dict | None = None) -> bool:
    user = user if user is not None else current_user()
    return bool(user) and user.get("role") in STAFF_ROLES


def can_view_student(student_id) -> bool:
    user = current_user()
    if not user:
        return False
    if user["role"] in STAFF_ROLES:
        return True
    return user["role"] == "student" and str(user.get("student_id")) == str(student_id)


def roles_required(*roles):
    """Decorator: require a logged-in user with one of `roles` (login redirect, or 403 if the role is wrong)."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if user is None:
                if session.get("username"):            # account was deleted while logged in
                    session.pop("username", None)
                return redirect(url_for("auth.login", next=request.path))
            if user["role"] not in roles:
                log_event("access_denied", request.path)
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


staff_required = roles_required(*STAFF_ROLES)
admin_required = roles_required("admin")


def login_required(view):
    return roles_required(*ROLES)(view)


# ── throttling ───────────────────────────────────────────────────────────────

def _recent_failures(key: str) -> list:
    window = current_app.config.get("LOGIN_WINDOW_SECONDS", 300)
    now = time.time()
    attempts = [t for t in _failed_logins.get(key, []) if now - t < window]
    _failed_logins[key] = attempts
    return attempts


def _safe_next(target: str | None) -> str | None:
    """Allow only same-site relative redirects (blocks open-redirect via ?next=)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return None


# ── routes ───────────────────────────────────────────────────────────────────

def _login_page(status=200):
    return render_template(
        "login.html",
        dev_hint=current_app.config.get("DEBUG") and DEV_ADMIN_USERNAME or None,
        dev_password=DEV_ADMIN_PASSWORD,
        no_accounts=not has_accounts(),
    ), status


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip_key = f"ip:{request.remote_addr}"
        username = request.form.get("username", "").strip()[:64]
        user_key = f"user:{username.lower()}"
        limit = current_app.config.get("LOGIN_MAX_ATTEMPTS", 5)
        if len(_recent_failures(ip_key)) >= limit or len(_recent_failures(user_key)) >= limit:
            log_event("login_blocked", username, user=username or "-")
            flash("Too many failed attempts. Please wait a few minutes and try again.", "danger")
            return _login_page(429)

        if verify_user(username, request.form.get("password", "")):
            _failed_logins.pop(ip_key, None)
            _failed_logins.pop(user_key, None)
            keep_csrf = session.get("_csrf_token")
            session.clear()                      # new session on login: no session fixation
            if keep_csrf:
                session["_csrf_token"] = keep_csrf
            session["username"] = username
            session.permanent = True             # idle timeout from PERMANENT_SESSION_LIFETIME
            log_event("login_ok", user=username)
            flash(f"Welcome back, {username}.", "success")
            user = get_user(username)
            default = url_for("students.my_report") if user["role"] == "student" else url_for("main.dashboard")
            return redirect(_safe_next(request.args.get("next")) or default)

        _failed_logins.setdefault(ip_key, []).append(time.time())
        _failed_logins.setdefault(user_key, []).append(time.time())
        log_event("login_failed", username, user=username or "-")
        flash("Invalid username or password.", "danger")

    return _login_page()


@auth_bp.route("/logout")
def logout():
    log_event("logout")
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("main.dashboard"))


@auth_bp.route("/password", methods=["GET", "POST"])
@login_required
def change_password():
    user = current_user()
    if request.method == "POST":
        old, new, again = (request.form.get(k, "") for k in ("old_password", "new_password", "confirm_password"))
        problem = validate_password(new)
        if not check_password_hash(user["password_hash"], old):
            flash("Your current password is not correct.", "danger")
        elif new != again:
            flash("The new passwords do not match.", "danger")
        elif problem:
            flash(problem, "danger")
        else:
            add_user(user["username"], new, role=user["role"], student_id=user.get("student_id"))
            log_event("password_changed")
            flash("Password changed.", "success")
            return redirect(url_for("main.dashboard"))
    return render_template("password.html")
