"""
Staff overview: Bayes' Theorem on the real records plus the top suggestions per student.
"""

from flask import Blueprint, render_template, request

from app.auth import staff_required
from app.data.analytics_calc import build_class_stats, get_all_recommendations
from app.data.data_loader import load_students
from app.ml.bayes import bayes_table

recommendations_bp = Blueprint("recommendations", __name__, url_prefix="/recommendations")

PAGE_SIZE = 20


@recommendations_bp.route("/")
@staff_required
def recommendations():
    df = load_students()
    if df.empty:
        return render_template("recommendations.html", rows=[], total=0, page=1, pages=1, only_risk=False,
                               bayes={"prior": None, "by_evidence": [], "worked_examples": []})

    only_risk = request.args.get("risk") == "1"
    view = df[df["at_risk"] == 1] if only_risk else df
    total = len(view)
    pages = max(1, -(-total // PAGE_SIZE))
    try:
        page = min(max(1, int(request.args.get("page", 1))), pages)
    except ValueError:
        page = 1
    chunk = view.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template(
        "recommendations.html", rows=get_all_recommendations(chunk, build_class_stats(df)),
        bayes=bayes_table(df), total=total, page=page, pages=pages, only_risk=only_risk,
    )
