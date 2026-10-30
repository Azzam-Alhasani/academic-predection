# Academic Performance Monitoring & Prediction System

A web application that monitors student academic records, predicts whether a student will pass, and gives simple, explainable recommendations. It runs on **real data**: the UCI *Student Performance* dataset (1,044 students from two Portuguese secondary schools). Built with Flask, Pandas and Scikit-learn.

---

## Features

- **Dashboard** — pass/fail counts, average grade, at-risk students, grade distribution, results by subject and by study time.
- **Student Management** — search, filter and page through students; add, edit and delete records (login required to change data).
- **Performance Analytics** — top/bottom performers, grade histogram, pass rate by study time / past failures / higher-education plans, absences vs grade.
- **ML Predictions** — five models with honest, cross-validated metrics (accuracy, precision, recall, F1, ROC-AUC, RMSE, R²) and a live "predict this student" form.
- **Bayesian Analysis** — Bayes' Theorem on the real records with the formula and a worked example.
- **Recommendations** — per-student trend, risk factors and plain-language advice.
- **Security** — CSRF protection, hashed-password login with throttling, input validation, safe cookies, HTTPS-ready.

---

## The data

| | |
|---|---|
| Source | P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, FUBUTEC 2008. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance), licence CC BY 4.0. |
| Records | 395 Mathematics + 649 Portuguese-language students = **1,044** |
| Target | `G3`, the final grade (0–20). **Pass = G3 ≥ 10** (78% of students pass). |
| Inputs | `G1`, `G2` (first- and second-period grades), study time, past failures, absences, age, gender, subject, wants higher education, home internet |

The raw files are in `app/data/raw/`. `python app/data/prepare_real_data.py` combines them into `app/data/sample_students.csv`. Student names are generic ("Student 0001"); the dataset is anonymised.

---

## ML models and results

Metrics come from **5-fold grouped cross-validation**: every student is scored by a model that never saw them, and a student who appears in both the Math and Portuguese files is always kept on the same side of the split, so there is no leakage. The final grade `G3` is never used as an input.

| Model | Task | Result (cross-validated) |
|---|---|---|
| Linear Regression | predict `G3` (0–20) | RMSE 1.55 grade points, R² 0.838 |
| Logistic Regression | pass / fail | accuracy **92.0%**, F1 0.949, ROC-AUC 0.971 |
| Random Forest | pass / fail | accuracy **92.0%**, F1 0.949, ROC-AUC 0.969 |
| Decision Tree | pass / fail | accuracy 90.9%, F1 0.943, ROC-AUC 0.949 |
| Naive Bayes | pass / fail | accuracy 86.6%, F1 0.914, ROC-AUC 0.926 |

**How to read this:** always guessing "pass" is right 78% of the time, so 92% is a real improvement. Several other model types and feature sets were tried (gradient boosting, all 30+ survey fields, tuned regularisation) and all land at 91–93%, so this is close to what the data supports. The models rely on the earlier period grades; **without `G1`/`G2` accuracy drops to about 80%**, barely above the baseline. That makes this an *early-warning* tool for mid-year, not a prediction made on day one.

Reproduce: `python -m app.ml.export_results` (rewrites `docs/model_results.json`).

---

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py                   # first start creates a local login:  admin / admin123
```

Open **http://127.0.0.1:5000** and log in with `admin` / `admin123` (this default account is created only on your own computer, never on a public server; change it with `python app/data/create_users.py admin YourNewPassword`). Or use the launchers (`run.bat` on Windows, `bash run.sh` on macOS/Linux), which do all of the above.

Run the tests with `pip install pytest && python -m pytest tests`.

---

## Put it on the public web (Render, free)

The repository includes `render.yaml`, so deployment is a few clicks:

1. Push this repository to GitHub.
2. Create an account at [render.com](https://render.com) and sign in with GitHub.
3. **New → Blueprint**, choose this repository.
4. When asked, enter a value for **`ADMIN_PASSWORD`** (your admin login; it is never stored in Git). `SECRET_KEY` is generated automatically.
5. Click **Apply**. After the build, Render gives you a public `https://…onrender.com` address.

Log in at `/auth/login` with username `admin` and the password you chose.

**Things to know about the free tier** (check Render's current limits):
- The site sleeps after a period without visitors; the first visit afterwards takes about a minute.
- The disk is temporary. Students added or edited through the website are lost when the service restarts or redeploys, and the data returns to the original dataset. For permanent edits, move storage to a database.
- Anyone on the internet can *view* the dashboard, student list, analytics and predictions; only logged-in users can add, edit or delete. That is fine for this anonymised public dataset. If you ever load real student records, make the read pages require login first.

Other hosts (Railway, Fly.io, a VPS) work too: set `FLASK_ENV=production`, `SECRET_KEY` and `ADMIN_PASSWORD`, and start with
`gunicorn run:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.

---

## Project structure

```
├── run.py                      Flask entry point (gunicorn uses run:app)
├── run.sh / run.bat            local one-click launchers
├── config.py                   development / production configuration
├── render.yaml                 one-click deployment on Render
├── requirements.txt
├── tests/test_app.py           automated tests
├── app/
│   ├── __init__.py             app factory, CSRF, proxy + security headers
│   ├── auth.py                 login, throttling, admin from ADMIN_PASSWORD
│   ├── security.py             CSRF tokens
│   ├── routes/                 dashboard, students, analytics, predictions, recommendations
│   ├── ml/
│   │   ├── preprocessing.py    features, scaling, grouped split
│   │   ├── trainer.py          5 models + cross-validated metrics
│   │   ├── predictor.py        model cache + single prediction
│   │   ├── bayes.py            hand-coded Bayes' Theorem
│   │   └── export_results.py   writes docs/model_results.json
│   ├── data/
│   │   ├── raw/                original UCI CSV files
│   │   ├── prepare_real_data.py builds sample_students.csv
│   │   ├── data_loader.py      read/write + derived columns
│   │   ├── analytics_calc.py   indicators + recommendations
│   │   └── create_users.py     create the admin login
│   ├── templates/              HTML pages
│   └── static/                 CSS, JS
└── docs/                       project documentation
```

---

## Security

| Area | Implementation |
|---|---|
| Authentication | hashed passwords; add/edit/delete and recommendations need a login |
| Brute force | 5 failed logins per IP in 5 minutes, then HTTP 429 |
| CSRF | token checked on every POST |
| Input validation | every form field checked against an allowed range / choice list |
| Cookies | HttpOnly, SameSite=Lax, Secure in production |
| Transport | HTTPS from the host; HSTS header and trusted-proxy handling in production |
| Secrets | `SECRET_KEY` is required in production; the admin password comes from an environment variable and is never committed |
| Redirects | `?next=` only accepts same-site paths |

Not covered (future work): a real database, roles beyond admin, rate limiting beyond login, audit logging.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| Blank charts | Allow access to the Bootstrap / Chart.js CDN, or host those files locally |
| `SECRET_KEY must be set` | Production only: set the `SECRET_KEY` environment variable |
| Can't log in (local) | Use `admin` / `admin123`, or reset with `python app/data/create_users.py admin NewPassword` |
| Can't log in (hosted) | Set `ADMIN_PASSWORD` in the host's environment settings and restart the service; username is `admin` |
| Edits disappeared on the hosted site | Free hosting has a temporary disk; see the hosting notes above |
| Port 5000 in use | Set the `PORT` environment variable |

---

## Tech stack

Python 3.10+, Flask, Pandas, NumPy, Scikit-learn, gunicorn, Bootstrap 5 and Chart.js (via CDN). Storage is CSV, so no database setup is needed.

---

## Licence

MIT. The dataset is © its authors, used under CC BY 4.0.
