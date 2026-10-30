import json

from flask import Blueprint, render_template, request

from app.auth import login_required
from app.data.analytics_calc import build_class_stats, get_all_recommendations
from app.data.data_loader import load_students
from app.ml.bayes import bayes_table

recommendations_bp = Blueprint("recommendations", __name__, url_prefix="/recommendations")

PAGE_SIZE = 20


@recommendations_bp.route("/")
@login_required
def recommendations():
    df = load_students()
    if df.empty:
        return render_template(
            "recommendations.html", rows=[], total=0, page=1, pages=1, only_risk=False,
            bayes={"prior_pass": None, "by_evidence": [], "worked_examples": []},
            class_stats={},
        )

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
        "recommendations.html",
        rows=get_all_recommendations(chunk),
        bayes=bayes_table(df),
        class_stats=build_class_stats(df),
        total=total, page=page, pages=pages, only_risk=only_risk,
    )
