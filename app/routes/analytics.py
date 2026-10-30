"""
Analytics route: class-level performance analysis and visualisation data.
"""

import json
from flask import Blueprint, render_template
from app.data.data_loader import get_analytics_data

analytics_bp = Blueprint("analytics", __name__, url_prefix="/analytics")


@analytics_bp.route("/")
def analytics():
    data = get_analytics_data()
    return render_template("analytics.html", data=data, data_json=json.dumps(data))
