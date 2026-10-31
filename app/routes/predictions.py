"""
Prediction tool and model-selection results.

The "try a student" calculator is public on purpose: it stores nothing, it only runs the
chosen model and the explainable calculations on the numbers typed in.
"""

import pandas as pd
from flask import Blueprint, render_template, request

from app import grading
from app.data.analytics_calc import build_class_stats, calculate_student_indicators, recommend_student
from app.data.data_loader import _compute_derived_columns, load_students
from app.ml.predictor import get_training_size, load_selection, predict_student
from app.ml.preprocessing import SCENARIOS
from app.routes.students import CHOICE_FIELDS, NUMBER_FIELDS, parse_student_form

predictions_bp = Blueprint("predictions", __name__, url_prefix="/predictions")

DEFAULTS = {
    "gender": "Female", "programme": "Informatics Engineering", "attendance": "Daytime",
    "scholarship": "no", "tuition_paid": "yes", "debtor": "no", "age": 19,
    "admission_mark": 63, "prev_mark": 62,
    "s1_enrolled": 6, "s1_approved": 5, "s1_mark": 62,
    "s2_enrolled": 0, "s2_approved": 0, "s2_mark": 0,
}


@predictions_bp.route("/", methods=["GET", "POST"])
def predictions():
    study = load_selection()
    form_data = dict(DEFAULTS)
    result = report = None
    error = None

    if request.method == "POST":
        submitted = request.form.to_dict()
        form_data.update({k: submitted.get(k, form_data.get(k)) for k in DEFAULTS})
        data, error = parse_student_form({**submitted, "outcome": "Enrolled", "name": ""})
        if not error:
            record = _compute_derived_columns(pd.DataFrame([{**data, "outcome": "Enrolled"}])).iloc[0].to_dict()
            record = {k: (None if isinstance(v, float) and v != v else v) for k, v in record.items()}
            result = predict_student(record)
            if "error" in result:
                error, result = result["error"], None
            else:
                df = load_students()
                class_stats = build_class_stats(df) if not df.empty else {}
                indicators = calculate_student_indicators(record, class_stats)
                report = {"indicators": indicators, "suggestions": recommend_student(record, class_stats, indicators, result),
                          "student": record}

    clf = study.get("classification", {})
    return render_template(
        "predictions.html", study=study, scenarios=SCENARIOS, clf=clf, reg=study.get("regression"),
        form=form_data, result=result, report=report, error=error, choices=CHOICE_FIELDS,
        number_fields=NUMBER_FIELDS, training_size=get_training_size(),
    )


@predictions_bp.route("/selection")
def model_selection():
    study = load_selection()
    return render_template("model_selection.html", study=study, scenarios=SCENARIOS,
                           scale=grading.scale_table())
