# Academic Performance Monitoring & Prediction System

A web application that tracks student results on a **UTAS-style grading scale** (letter grades, GPA, academic standing), estimates the chance that a student **fails a course**, and gives each student explainable calculations and suggestions. Built with Flask, Pandas and Scikit-learn.

It looks at only what drives results: **term marks, absences and the final mark**. No age, gender, family, scholarship or fee data is used.

> **Please read: what is real and what is assumed**
> - **Data.** Real records from the UCI *Student Performance* dataset (two Portuguese secondary-school courses, Mathematics and Portuguese, 1,044 course records of 662 students, CC BY 4.0). It is **not UTAS data**. Marks (0-20) were converted to a 0-100 scale, so results show how the system behaves, not how UTAS students perform.
> - **Grading scale.** The letter bands, grade points and standing thresholds in `app/grading.py` are **assumptions** based on a typical Omani 100-mark scale. No official UTAS table was available. Check them against the UTAS handbook and edit that one file; everything else follows.
> - **Absences** are a count of classes missed, as recorded in the dataset. There is no total number of classes, so no attendance percentage is shown.

---

## Features

- **Dashboard and analytics**: pass/fail, letter grades, chance of failing by term 1 mark, GPA, change between terms, absences and course (aggregates only).
- **Student report** (staff, or the student for their own record): term results, GPA, average, trend, safety margin above the pass mark, peer percentile, absences, a **goal planner** (the final mark needed for a target average) and a printable report. Every figure shows its formula and working.
- **Suggestions** ranked by priority, each with the reason and the calculation behind it.
- **Risk calculator** (public, stores nothing): enter term 1 (and optionally term 2) and absences; get the estimated risk of failing, the factors raising and lowering it, a projected final mark, and the report.
- **Model-selection analysis**: how the model was chosen and how accurate it is (below).
- **Bayesian analysis**: P(Fail | evidence) from the real records, with worked examples.
- **Roles**: admin, teacher, student. **Audit log** of logins and data changes.

---

## UTAS-style grading (`app/grading.py`)

| Mark | Grade | Points | | Mark | Grade | Points |
|---|---|---|---|---|---|---|
| 90-100 | A+ | 4.0 | | 65-69 | C | 2.0 |
| 85-89 | A | 4.0 | | 60-64 | D+ | 1.5 |
| 80-84 | B+ | 3.5 | | 50-59 | D | 1.0 |
| 75-79 | B | 3.0 | | 0-49 | F | 0.0 |
| 70-74 | C+ | 2.5 | | | | |

Pass mark 50. **Term GPA** = grade points of the term mark. **GPA** = average of the term GPAs. Standing (assumed, scaled to the data): Good standing from 1.0, Academic warning from 0.5, otherwise Academic probation. In the data, GPA 1.0 or more almost never fails (about 1%); 0.5-1.0 fails about 3 in 10; under 0.5 about 8 in 10. With real UTAS marks use the handbook thresholds.

---

## The data

| | |
|---|---|
| Source | P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, FUBUTEC 2008. [UCI #320](https://archive.ics.uci.edu/dataset/320/student+performance), CC BY 4.0 |
| Records | 395 Mathematics + 649 Portuguese = 1,044 course records; 382 students appear in both |
| Kept | course, term 1 mark (`G1`), term 2 mark (`G2`), absences, final mark (`G3`) |
| Dropped | everything else in the original files (age, sex, family, free time, alcohol ...) |
| Conversion | mark out of 20 x 5 = mark out of 100. **Fail = final mark below 50** (22% of records) |

`python app/data/prepare_real_data.py` builds `app/data/sample_students.csv`. A hidden `person_id` links one student's two courses so both always stay on the same side of every train/test split (no leakage). Names are generated ("Student 0001").

---

## How the ML model was chosen

The choice is a documented experiment (`python -m app.ml.model_selection`, about 2 minutes; results in `app/ml/model_selection.json`, `docs/09_model_selection.md` and the **Model selection** page).

1. **10 classifiers** (baseline, logistic, decision tree, k-NN, naive Bayes, SVM, random forest, extra trees, gradient boosting, neural network) scored with the same 5-fold cross-validation repeated 3 times, grouped by student.
2. **Nested cross-validation** to tune the best ones, so tuning cannot inflate the score.
3. **Corrected resampled t-tests** (Nadeau and Bengio) to see whether differences are real, then a **parsimony rule** (simplest model within one standard error of the best).
4. For the chosen model: threshold analysis, calibration, permutation importance and a learning curve.

| Question | Selected model | Accuracy | ROC-AUC | Failing students caught | "Always say passes" |
|---|---|---|---|---|---|
| Will the student fail? After term 1 | Gradient boosting | **89.2%** | 0.945 | 73% | 78.0% |
| After term 2 | Logistic regression | **92.3%** | 0.974 | 79% | 78.0% |
| Final mark (regression) | Random forest | within 5 marks: 52% / 75%; within 10: 79% / 93% (after term 1 / 2) | | | |

**Honest reading.**
- Accuracy is above the 85% target at both moments, and clearly above the 78% you get by always guessing "passes".
- **The term marks do almost all the work.** Absences and course add very little once the marks are known (see the importance chart). Absences matter mainly at the extreme: 36% of records with 11+ absences failed, against 17-23% for the others.
- After term 2, logistic regression, gradient boosting and random forest are statistically tied, so the simplest, explainable one is used.
- About 27% of students who later fail are still missed after term 1 at the default 50% alert level; lowering it to 40% catches about 80% with similar accuracy.
- The numbers are for this Portuguese data; retrain on UTAS records before relying on them.

---

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py                   # first start creates a local login:  admin / admin123
python -m pytest tests          # optional: 21 tests
```

Open **http://127.0.0.1:5000** and log in with `admin` / `admin123` (created only on your own computer, never on a public server). Change it on the *Password* page. Or use `run.bat` (Windows) / `bash run.sh`.

Add teacher and student accounts under **Accounts** (admin). A student account is linked to a student ID and sees only that student's report.

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
- Anyone on the internet can see the dashboard, analytics, risk calculator and model-selection pages (aggregates only, no names). Student lists, reports, recommendations and admin pages need a login. Before loading **real UTAS records**, keep the site private (login on every page) and use a database.

Other hosts (Railway, Fly.io, a VPS) work too: set `FLASK_ENV=production`, `SECRET_KEY` and `ADMIN_PASSWORD`, and start with
`gunicorn run:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.

---

## Project structure

```
├── run.py, run.sh, run.bat, config.py, render.yaml, requirements.txt
├── tests/test_app.py             21 automated tests
├── app/
│   ├── grading.py                UTAS-style scale, GPA, planner formulas (edit here)
│   ├── auth.py, security.py, audit.py   roles, login, CSRF, audit log
│   ├── routes/                   dashboard, students, analytics, predictions, recommendations, admin
│   ├── ml/                       preprocessing, registry (models), model_selection (study), predictor, bayes
│   ├── data/                     raw data, prepare_real_data.py, data_loader.py, analytics_calc.py
│   ├── templates/, static/
└── docs/                         documentation (08 security, 09 model selection)
```

---

## Security (reviewer comment: "Local server is used, reducing security")

The system is now designed to run on a public HTTPS host (Render) instead of a laptop. Details in `docs/08_security_architecture.md`.

| Area | Implementation |
|---|---|
| Hosting | HTTPS from the host, HSTS, trusted-proxy handling, health check |
| Access control | roles admin / teacher / student, enforced on the server; students see only their own record |
| Passwords | hashed; 10+ characters in production, letters and digits, common passwords refused |
| Brute force | lockout by IP and by username; equal timing for unknown users |
| Sessions | cleared on login (no fixation), 30-minute idle timeout, HttpOnly, SameSite, Secure |
| CSRF | token checked on every POST (constant-time compare) |
| Browser protection | Content-Security-Policy with per-request nonce, no inline scripts, X-Frame-Options, nosniff, `no-store` for logged-in pages |
| Input | whitelist validation, size limit (64 KB), spreadsheet-formula characters stripped from names |
| Audit | logins, failures, data changes (`app/data/audit.log`, admin page) |
| Secrets | `SECRET_KEY` and `ADMIN_PASSWORD` come from the environment, never from Git |

**Not covered (future work):** a real database (data is CSV and Render's free disk resets), multi-factor login, per-user password reset by email, an external penetration test.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| Blank charts | Allow access to the Bootstrap / Chart.js CDN, or host those files locally |
| `SECRET_KEY must be set` | Production only: set the `SECRET_KEY` environment variable |
| Can't log in (local) | Use `admin` / `admin123`, or reset with `python app/data/create_users.py NewPassword` |
| Can't log in (hosted) | Set `ADMIN_PASSWORD` in the host's environment settings and restart the service; username is `admin` |
| Edits disappeared on the hosted site | Free hosting has a temporary disk; see the hosting notes above |
| Port 5000 in use | Set the `PORT` environment variable |

---

## Tech stack

Python 3.10+, Flask, Pandas, NumPy, Scikit-learn, gunicorn, Bootstrap 5 and Chart.js (CDN). Storage is CSV.

---

## Licence

MIT. The dataset is (c) its authors, used under CC BY 4.0.
