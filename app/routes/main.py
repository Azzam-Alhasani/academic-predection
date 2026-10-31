"""
Main dashboard route (aggregate statistics only: safe to show without a login).
"""

from flask import Blueprint, render_template

from app.data.data_loader import get_dashboard_stats

main_bp = Blueprint("main", __name__)


@main_bp.route("/healthz")
def healthz():
    """Lightweight check for the hosting platform (does not load data or train models)."""
    return {"status": "ok"}


@main_bp.route("/")
def dashboard():
    return render_template("dashboard.html", stats=get_dashboard_stats())
