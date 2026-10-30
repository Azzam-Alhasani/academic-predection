"""
Simple multi-user authentication. Accounts are stored in `app/data/users.csv`
with hashed passwords; sessions use Flask's signed cookie.
"""

import os
import time
import pandas as pd
from functools import wraps
from flask import (
    Blueprint, render_template, request, redirect, url_for, session, flash, current_app,
)
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Failed login attempts per client IP: {ip: [timestamps]}. In-memory is enough for a
# single-process deployment and keeps brute-force guessing slow.
_failed_logins: dict = {}


def _client_ip() -> str:
    # remote_addr is the real client address: in production create_app() wraps the app in
    # ProxyFix, which takes it from the host's trusted proxy. Reading X-Forwarded-For
    # directly here would let anyone dodge the throttle by sending a fake header.
    return request.remote_addr or "unknown"


def _recent_failures(ip: str) -> list:
    window = current_app.config.get("LOGIN_WINDOW_SECONDS", 300)
    now = time.time()
    attempts = [t for t in _failed_logins.get(ip, []) if now - t < window]
    _failed_logins[ip] = attempts
    return attempts


def _safe_next(target: str | None) -> str | None:
    """Allow only same-site relative redirects (blocks open-redirect via ?next=)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return None


DEV_ADMIN_USERNAME = "admin"
DEV_ADMIN_PASSWORD = "admin123"


def has_accounts() -> bool:
    return not load_users().empty


def ensure_dev_admin(app) -> None:
    """
    Development only: when no account exists yet, create admin / admin123 so a fresh
    install can always log in. Never runs in production, where ADMIN_PASSWORD is required.
    """
    if app.config.get("DEBUG") and not has_accounts():
        add_user(DEV_ADMIN_USERNAME, DEV_ADMIN_PASSWORD, role="admin")
        print(f" * First run: created the development login  {DEV_ADMIN_USERNAME} / {DEV_ADMIN_PASSWORD}")


def ensure_admin_from_env() -> None:
    """
    Create the admin account from environment variables when none exists yet.
    Used on hosts with no shell access (e.g. Render): set ADMIN_PASSWORD (and
    optionally ADMIN_USERNAME) and the account is created at start-up.
    """
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        return
    username = os.environ.get("ADMIN_USERNAME", "admin")
    df = load_users()
    if df.empty or not (df["username"] == username).any():
        add_user(username, password, role="admin")


def _users_file():
    # Use __file__ (not current_app.root_path) so this also works when
    # create_users.py runs standalone without a Flask app context.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.csv")


def load_users() -> pd.DataFrame:
    """Load users from CSV (empty DataFrame if not found)."""
    path = _users_file()
    if not os.path.exists(path):
        return pd.DataFrame(columns=["username", "password_hash", "role"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["username", "password_hash", "role"])


def _save_users(df: pd.DataFrame) -> None:
    df.to_csv(_users_file(), index=False)


def get_user(username: str) -> dict | None:
    """Return a user record dict, or None if not found."""
    df = load_users()
    if df.empty:
        return None
    rows = df[df["username"] == username]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def verify_user(username: str, password: str) -> bool:
    """Check a username/password pair against the stored hash."""
    user = get_user(username)
    if user is None:
        return False
    return check_password_hash(user["password_hash"], password)


def current_user() -> dict | None:
    """Return the currently logged-in user, or None."""
    username = session.get("username")
    if not username:
        return None
    return get_user(username)


def login_required(view):
    """Decorator: redirect to login if no user is in the session."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("username") is None:
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def add_user(username: str, password: str, role: str = "admin") -> None:
    """Create or replace a user account with a hashed password."""
    df = load_users()
    password_hash = generate_password_hash(password)
    if not df.empty and (df["username"] == username).any():
        df.loc[df["username"] == username, "password_hash"] = password_hash
        df.loc[df["username"] == username, "role"] = role
    else:
        new_row = pd.DataFrame([{"username": username, "password_hash": password_hash, "role": role}])
        df = pd.concat([df, new_row], ignore_index=True)
    _save_users(df)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ip = _client_ip()
        limit = current_app.config.get("LOGIN_MAX_ATTEMPTS", 5)
        if len(_recent_failures(ip)) >= limit:
            flash("Too many failed attempts. Please wait a few minutes and try again.", "danger")
            return render_template("login.html", no_accounts=False), 429

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_user(username, password):
            _failed_logins.pop(ip, None)
            session["username"] = username
            flash(f"Welcome back, {username}.", "success")
            return redirect(_safe_next(request.args.get("next")) or url_for("main.dashboard"))
        _failed_logins.setdefault(ip, []).append(time.time())
        flash("Invalid username or password.", "danger")

    return render_template(
        "login.html",
        dev_hint=current_app.config.get("DEBUG") and DEV_ADMIN_USERNAME or None,
        dev_password=DEV_ADMIN_PASSWORD,
        no_accounts=not has_accounts(),
    )


@auth_bp.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("main.dashboard"))