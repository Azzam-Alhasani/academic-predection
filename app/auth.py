"""
Simple multi-user authentication. Accounts are stored in `app/data/users.csv`
with hashed passwords; sessions use Flask's signed cookie.
"""

import os
import pandas as pd
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


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
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_user(username, password):
            session["username"] = username
            flash(f"Welcome back, {username}.", "success")
            return redirect(request.args.get("next") or url_for("main.dashboard"))
        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("main.dashboard"))