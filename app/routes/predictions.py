"""
Predictions route: run the ML models for one student and show cross-validated metrics.
"""

import json

from flask import Blueprint, flash, render_template, request

from app.data.data_loader import load_students
from app.ml.predictor import get_model_metrics, get_training_size, predict_student

predictions_bp = Blueprint("predictions", __name__, url_prefix="/predictions")

# Inputs shown on the prediction form: (field, minimum, maximum)
NUMERIC_INPUTS = {
    "age": (10, 30),
    "studytime": (1, 4),
    "failures": (0, 4),
    "absences": (0, 100),
    "G1": (0, 20),
    "G2": (0, 20),
}
CHOICES = {
    "subject": ("Math", "Portuguese"),
    "gender": ("Male", "Female"),
    "higher": ("yes", "no"),
    "internet": ("yes", "no"),
}
DEFAULTS = {
    "subject": "Math", "gender": "Female", "higher": "yes", "internet": "yes",
    "age": 17, "studytime": 2, "failures": 0, "absences": 4, "G1": 10, "G2": 10,
}


@predictions_bp.route("/", methods=["GET", "POST"])
def predictions():
    metrics = get_model_metrics()
    prediction_results = None
    form_data = dict(DEFAULTS)
    selected_student = None

    # Dropdown of existing students (id, name, subject only: keeps the page light)
    df = load_students()
    students = []
    if not df.empty:
        students = df[["student_id", "name", "subject"]].to_dict("records")

    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()

        if student_id and not df.empty:
            try:
                sid = int(student_id)
            except ValueError:
                sid = -1
            row = df[df["student_id"] == sid]
            if row.empty:
                flash("Student not found.", "warning")
            else:
                selected_student = row.iloc[0].to_dict()
                form_data = {k: selected_student.get(k, v) for k, v in DEFAULTS.items()}
                form_data["student_id"] = sid
        else:
            form_data = _read_custom_form(request.form)

        if selected_student is not None or request.form.get("mode") != "student":
            raw = predict_student(form_data)
            if "error" in raw:
                flash(raw["error"], "warning")
            else:
                prediction_results = raw

    return render_template(
        "predictions.html",
        metrics=metrics,
        metrics_json=json.dumps(metrics),
        prediction_results=prediction_results,
        form_data=form_data,
        students=students,
        selected_student=selected_student,
        training_size=get_training_size(),
        choices=CHOICES,
        ranges=NUMERIC_INPUTS,
    )


def _read_custom_form(form) -> dict:
    """Read the hypothetical-student form, clamping every value to its valid range."""
    data = {}
    for key, allowed in CHOICES.items():
        value = form.get(key, DEFAULTS[key])
        data[key] = value if value in allowed else DEFAULTS[key]
    for key, (lo, hi) in NUMERIC_INPUTS.items():
        data[key] = _clamp(form.get(key), lo, hi, DEFAULTS[key])
    return data


def _clamp(value, lo, hi, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return int(min(hi, max(lo, number)))
