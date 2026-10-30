"""
Main dashboard route.
"""

import json
from flask import Blueprint, render_template
from app.data.data_loader import get_dashboard_stats

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats, stats_json=json.dumps(stats))
