"""
Analytics route: class-level performance analysis (aggregates only).
"""

from flask import Blueprint, render_template

from app.data.data_loader import get_analytics_data

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")


@analytics_bp.route("/")
def analytics():
    return render_template("analytics.html", data=get_analytics_data())
