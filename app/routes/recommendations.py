import json
from flask import Blueprint, render_template
from app.data.data_loader import load_students
from app.data.analytics_calc import get_all_recommendations, build_class_stats
from app.ml.bayes import bayes_table
from app.auth import login_required

recommendations_bp = Blueprint("recommendations", __name__, url_prefix="/recommendations")


@recommendations_bp.route("/")
@login_required
def recommendations():
    df = load_students()
    if df.empty:
        return render_template(
            "recommendations.html",
            rows=[],
            bayes={"prior_pass": None, "by_evidence": [], "worked_examples": []},
            class_stats={},
            rows_json=json.dumps([]),
        )

    rows = get_all_recommendations(df)
    bayes = bayes_table(df)
    class_stats = build_class_stats(df)
    return render_template(
        "recommendations.html",
        rows=rows,
        bayes=bayes,
        class_stats=class_stats,
        rows_json=json.dumps(rows),
    )