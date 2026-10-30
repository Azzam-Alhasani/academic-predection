# Academic Performance Monitoring & Prediction System

A student web application for monitoring academic records, predicting performance with machine learning, and giving each student simple, explainable recommendations. Built with Flask, Pandas, and Scikit-learn with a dark dashboard UI.

---

## Features

- **Dashboard** — KPI cards (students, pass/fail, GPA, at-risk), grade distribution, subject averages, study-hours trend, and class improvement summary.
- **Student Management** — add, view, edit, and delete student records through forms.
- **Performance Analytics** — top/bottom performers, GPA histogram, gender comparison, attendance scatter plot.
- **ML Predictions** — five models with live evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, RMSE, R²) and a confusion matrix.
- **Bayesian Analysis** — Bayes' Theorem applied to pass/fail using the actual student records, with the full formula shown and a worked example.
- **Recommendations** — per-student average, improvement/decline trend, weak subjects, and plain-language advice.
- **At-Risk Detection** — automatic flagging of students with low scores or attendance.
- **Local Server Security** — CSRF protection, login authentication, input validation, and safe localhost defaults.
- **Sample Data** — 500 demo students generated automatically.

---

## Quick Start

### Option A — One-click launcher

- **Windows:** double-click `run.bat`.
- **macOS / Linux:** run `bash run.sh`.

These create a virtual environment, install dependencies, generate sample data, and start the app.

### Option B — Manual setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# .\venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample data (500 students)
python app/data/generate_sample_data.py

# 4. Create the admin login (prints a password)
python app/data/create_users.py

# 5. Run the app
python run.py
```

Open **http://127.0.0.1:5000** in your browser.

> **Note:** `python-dotenv` is optional; `SECRET_KEY` and `FLASK_ENV` can be set as environment variables or in a `.env` file.

---

## Login

Student data is protected by a simple login. Add/edit/delete actions and the recommendations page require an account.

```bash
python app/data/create_users.py           # default user "admin", random password
python app/data/create_users.py admin123  # set a specific password
```

The password hash is stored in `app/data/users.csv` (the plain password is never saved).

---

## Project Structure

```
├── run.py                          Flask entry point
├── run.sh / run.bat                one-click launchers
├── config.py                       configuration (dev/prod)
├── requirements.txt
├── app/
│   ├── __init__.py                 app factory, CSRF hook, template helpers
│   ├── auth.py                     login, logout, login_required, user store
│   ├── security.py                 CSRF token generation & validation
│   ├── routes/
│   │   ├── main.py                 dashboard
│   │   ├── students.py             student CRUD + validation
│   │   ├── analytics.py            class analytics
│   │   ├── predictions.py          ML predictions + metrics
│   │   └── recommendations.py      recommendations + Bayes table
│   ├── ml/
│   │   ├── preprocessing.py        feature encoding, scaling, split
│   │   ├── trainer.py              trains all 5 models + metrics
│   │   ├── predictor.py            model cache + single prediction
│   │   └── bayes.py                hand-coded Bayes' Theorem table
│   ├── data/
│   │   ├── data_loader.py          CSV read/write + derived columns
│   │   ├── analytics_calc.py       indicators + recommendations
│   │   ├── generate_sample_data.py sample data generator
│   │   ├── create_users.py         seed login accounts
│   │   ├── students.csv            working data
│   │   └── sample_students.csv     fresh sample (500 records)
│   ├── templates/                  HTML pages
│   └── static/                     CSS, JS
└── docs/                           detailed documentation
```

---

## ML Models & Evaluation

| Model               | Task           | Target            |
|---------------------|----------------|-------------------|
| Linear Regression   | Regression     | `avg_score` (0–100)|
| Logistic Regression | Classification | `pass_fail` (0/1)  |
| Random Forest       | Classification | `pass_fail` (0/1)  |
| Decision Tree       | Classification | `pass_fail` (0/1)  |
| Naive Bayes         | Classification | `pass_fail` (0/1)  |

**Metrics computed automatically** (on the page and in `docs/model_results.json`):

- Regression: RMSE, R².
- Classification: accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC (reported where the model supports it).

**Actual sample-data results** (recorded in `docs/model_results.json`):

| Model               | Result |
|---------------------|--------|
| Linear Regression   | RMSE 5.358, R² 0.754 |
| Logistic Regression | accuracy 0.97, precision 0.98, recall 0.99, F1 0.985 |
| Random Forest       | accuracy 0.96, precision 0.98, recall 0.98, F1 0.980 |
| Decision Tree       | accuracy 0.95, precision 0.989, recall 0.959, F1 0.974 |

Naive Bayes metrics are calculated live on first run (no pre-saved values are invented).

---

## Bayesian Analysis

The **Recommendations** page shows Bayes' Theorem computed from the actual records, not from the ML model:

```
P(Pass | Evidence) = P(Evidence | Pass) × P(Pass) / P(Evidence)
```

The page displays:

1. **P(Pass)** — the base rate of passing across the class.
2. A table of evidence (previous-term outcome and attendance band) with `P(Evidence | Pass)`, `P(Evidence)`, and `P(Pass | Evidence)`.
3. A worked example with the exact numbers plugged into the formula.

This makes the theorem easy to trace and compare against the Naive Bayes model, which learns these probabilities automatically from the features.

---

## Calculations & Recommendations

Simple, explainable formulas (`app/data/analytics_calc.py`):

| Indicator | Formula |
|-----------|---------|
| Average score | mean of the 5 subject scores |
| Improvement | current average − previous term average |
| Trend | Improving (> +2), Declining (< −2), Stable |
| Weak subjects | any subject below 60 |
| Strongest subject | subject with the highest score |
| Assignments gap | 12 − assignments submitted |

Based on these, the system generates personalized advice such as:

- *"Your average dropped 8.4 points compared to last term. Meet your instructor to review where the decline is coming from."*
- *"Focus on your weakest subject(s): Math. Spend extra time on practice questions and ask for help early."*
- *"Attendance is 55%, below the 60% safe level. Attending classes more regularly is the single biggest improvement you can make."*

---

## Security

| Area | Implementation |
|------|----------------|
| Input validation | `students.py` validates every field against an allowed range |
| CSRF | hidden token checked on every POST (`security.py`) |
| Authentication | hashed passwords + login session (`auth.py`) |
| Secret key | required in production; dev fallback only in development |
| Binding | server binds to `127.0.0.1` by default (not exposed on the network) |

**Essential now:** input validation, CSRF, authentication, localhost binding.

**Left for future deployment:** HTTPS (reverse proxy / TLS), rate limiting, a real database (instead of CSV), role-based permissions (guest/teacher/admin), and moving `SECRET_KEY` into a secrets manager.

---

## Pages

| URL               | Description |
|-------------------|-------------|
| `/`               | Dashboard |
| `/students`       | Student list (search, add, delete) |
| `/students/<id>`  | Student profile with trend, weak subjects, recommendations |
| `/analytics`      | Class-wide analytics |
| `/predictions`    | Run all models + evaluation metrics |
| `/recommendations`| Bayes table + per-student recommendations |
| `/auth/login`     | Login |

---

## Documentation

More detail lives in [`docs/`](docs/):

1. [Project Overview](docs/01_project_overview.md)
2. [Data Collection & Management](docs/02_data_collection_management.md)
3. [Performance Analysis](docs/03_performance_analysis.md)
4. [ML Prediction Models](docs/04_ml_prediction_models.md)
5. [Deployment Guide](docs/05_deployment_guide.md)
6. [Model Metrics Comparison](docs/06_model_metrics_comparison.md)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| Blank charts | Allow CDN access (Bootstrap/Chart.js) or host the assets locally |
| "Not enough data" | Add at least 10 students |
| Port 5000 in use | Set the `PORT` environment variable or edit `config.py` |
| `SECRET_KEY must be set` | Set `SECRET_KEY` (production only) |
| Windows venv on macOS | Recreate the venv with `python3 -m venv venv` |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.10+, Flask |
| Data | Pandas, NumPy |
| ML | Scikit-learn |
| Frontend | Bootstrap 5 (dark), Chart.js, Font Awesome |
| Storage | CSV (no database required) |

---

## What DeepSeek Changed

This section documents the changes made to add the three requested requirements.

**1. Machine Learning Model Analysis**
- `app/ml/trainer.py` — added **Naive Bayes** as a fifth model, plus **ROC-AUC** and kept the **confusion matrix**. Added `_clf_metrics()` so accuracy/precision/recall/F1/ROC-AUC are all computed consistently.
- `app/ml/bayes.py` *(new)* — hand-coded **Bayes' Theorem** applied to pass/fail using previous-term outcome and attendance bands, with a worked example (separate from the sklearn model).
- `app/routes/recommendations.py` *(new)* — renders the Bayes table and recommendations.
- `app/templates/recommendations.html` *(new)* — page showing the prior, evidence table, and worked example.
- `app/templates/predictions.html` — shows ROC-AUC and the Naive Bayes model.

**2. Local Server Security**
- `app/security.py` *(new)* — lightweight CSRF token generation and validation.
- `app/auth.py` *(new)* — login/logout with hashed passwords and a `login_required` decorator.
- `app/data/create_users.py` *(new)* — seeds an admin account.
- `app/templates/login.html` *(new)* — login form.
- `app/__init__.py` — registered the CSRF `before_request` hook, auth blueprint, recommendations blueprint, and template helpers.
- `config.py` — split into `DevelopmentConfig` / `ProductionConfig`; requires `SECRET_KEY` in production; defaults to localhost.
- `run.py` — binds to `127.0.0.1` by default.
- `app/routes/students.py` — added numeric range validation for all form fields.

**3. Additional Calculations & Recommendations**
- `app/data/analytics_calc.py` *(new)* — average, improvement/decline, weak subjects, strongest subject, and a rule-based recommendation engine.
- `app/data/data_loader.py` — added `previous_term_avg` baseline and class-level improvement counts for the dashboard.
- `app/data/generate_sample_data.py` — added `previous_term_avg` to the sample data.
- `app/templates/student_detail.html` — added trend/weak-subject/recommendation cards and a "Previous Term Avg" field.
- `app/templates/dashboard.html` — added class improvement summary cards.
- `app/templates/students.html` — added a "Previous Term Avg" field to the add form.

**Other**
- Cleaned up verbose comments/docstrings in the Python modules so the code stays readable and focused.
- Regenerated `app/data/sample_students.csv` (and `students.csv`) to include `previous_term_avg`.

---

## License

MIT License — free to use, modify, and distribute.