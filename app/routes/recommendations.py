"""
Staff overview: Bayes' Theorem on the real records plus the top suggestions per student.
"""

from flask import Blueprint, render_template, request

from app.auth import staff_required
from app.data import data_loader
from app.data.analytics_calc import plan_for_student
from app.ml.bayes import bayes_table
from app.ml.predictor import get_frame

recommendations_bp = Blueprint("recommendations", __name__, url_prefix="/recommendations")

PAGE_SIZE = 12


@recommendations_bp.route("/")
@staff_required
def recommendations():
    t = data_loader.load_tables()
    if t.empty:
        return render_template("recommendations.html", rows=[], total=0, page=1, pages=1, only_risk=False,
                               bayes={"prior": None, "by_evidence": [], "worked_examples": []})
    only_risk = request.args.get("risk") == "1"
    view = t.summary[t.summary["current_courses"] > 0]
    if only_risk:
        view = view[view["at_risk"] == 1]
    total = len(view)
    pages = max(1, -(-total // PAGE_SIZE))
    try:
        page = min(max(1, int(request.args.get("page", 1))), pages)
    except ValueError:
        page = 1
    chunk = view.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    rows = []
    for rec in chunk.to_dict("records"):
        plan = plan_for_student(int(rec["student_id"]))
        rows.append({"student": rec, "plan": plan, "top": (plan.get("suggestions") or [])[:3]})
    tiers = {r.course_code: r.tier for r in data_loader.course_stats().itertuples()}
    return render_template("recommendations.html", rows=rows, bayes=bayes_table(get_frame(), tiers), total=total,
                           page=page, pages=pages, only_risk=only_risk)
