# Academic Performance Monitoring & Prediction System

A web application built around a **UTAS-style transcript**: semesters, courses, credit hours, marks, attendance %, W withdrawals, repeats, SGPA / CGPA / LCGPA and academic standing. For the courses a student is taking now it estimates the mark and the chance of failing each one, shows **which course is hardest compared with the others**, and answers two what-ifs: *"Would withdrawing one course (say Calculus II or Physics II) raise my GPA?"* and *"What if I attended more?"*. Built with Flask, Pandas and Scikit-learn.

Only marks and attendance drive the predictions. Gender is stored for reporting and is **not** a model input. There is no age, scholarship or fee data.

> **Please read: what is real and what is assumed**
> - **The data is simulated.** No public dataset has this transcript structure, and real transcripts cannot be published. `app/data/generate_students.py` creates 850 synthetic students in 5 majors (Engineering, Business, Information Technology, Applied Sciences, Design), 117 courses and about 17,000 course records. It is **modelled on the layout** of a UTAS transcript and contains no real student's data. The attendance effect, course difficulty and noise come from my assumptions, so accuracy figures show that the pipeline works, **not** how accurate it would be on real UTAS students.
> - **From the transcript:** the letter-to-grade-point table (A 4.0 ... F 0.0), W courses not counted, and a pass needing C or better (C- and D are fails).
> - **Assumed (edit `app/grading.py`):** the mark ranges for each letter (A from 90, A- 85, B+ 80 ... C 60 ... F below 45), the pass mark of 60, standing (Good from CGPA 2.0, Warning from 1.5, otherwise Probation) and credit-load limits (light under 12, normal 12-18, overload above 18). LCGPA is implemented as "CGPA counting only the latest attempt of a repeated course", which is my reading of the transcript.
> - **Advice is statistical.** A recommendation to withdraw or attend more is an estimate from the model, not a promise. Always check withdrawal deadlines with your advisor.

---

## Features

- **Semester planner** (public, stores nothing): pick a major, enter your CGPA, credits so far and the courses you take with your attendance. You get the expected mark, grade and chance of failing per course, a Hard / Medium / Easy difficulty tier compared with the other courses, SGPA and CGPA forecasts, the **withdraw one course** table, the **attend more** what-if (at 90%), a target-CGPA calculator, ranked recommendations and the formulas with working.
- **Student page** (staff, or the student for their own record): the transcript laid out like the UTAS one (per-semester SGPA, LCGPA, CGPA, credit load), the plan for the current semester, a goal planner, a printable report, and for staff add / edit / delete of students and course records.
- **Dashboard and analytics** (aggregates only): grade distribution, CGPA by major, failure rate by attendance band, major, credit load, year, gender and course difficulty tier; the hardest and easiest courses.
- **Course difficulty** is measured from the records: how many marks lower students score in a course than in their other courses (average over attempts), split into terciles.
- **Model-selection analysis** and **Bayesian analysis** (P(Fail | evidence) from the records).
- **Roles**: admin, teacher, student. **Audit log** of logins and data changes.

### How the withdraw and attendance advice works
1. For each course the models give the expected mark and the chance of an F.
2. Withdrawing a course removes its credits from the semester; SGPA and CGPA are recomputed without it. A course is suggested only if its chance of failing is at least 30% and CGPA rises by at least 0.03.
3. If attendance is below 85% and the chance of failing at 90% attendance would fall under 25%, the advice is **"fix attendance first"**; withdrawal is the fallback.
4. Attending more is never predicted to lower a mark. Dropping below 12 credits triggers a warning.

---

## Transcript rules (`app/grading.py`)

| Grade | Points | | Grade | Points | | Grade | Points |
|---|---|---|---|---|---|---|---|
| A | 4.0 | | B | 3.0 | | C- | 1.7 (fail) |
| A- | 3.7 | | B- | 2.7 | | D+ | 1.3 (fail) |
| B+ | 3.3 | | C+ | 2.3 | | D | 1.0 (fail) |
| | | | C | 2.0 (pass) | | F | 0.0 (fail) |

Quality points = credits x grade points. **SGPA** = quality points / attempted credits that semester. **CGPA** = the same over all attempts. **LCGPA** counts only the latest attempt of each course. A **W** course is not attempted and has no points. Needed SGPA for a target = (target x (A + n) - CGPA x A) / n, with A credits so far and n credits this semester.

---

## The data

| File (`app/data/sample/`) | Columns |
|---|---|
| `students.csv` | `student_id, name, gender, major, entry_semester` |
| `courses.csv` | `course_code, course_name, major, credit_hours, level` |
| `enrollments.csv` | `enrollment_id, student_id, semester, course_code, mark, attendance, status` (status: Completed, Withdrawn or In progress; mark empty unless Completed) |

Regenerate with `python app/data/generate_students.py`. **To use real data**, replace these three files with anonymised transcripts in the same format (no real names or student numbers), then run `python -m app.ml.model_selection` to redo the model study. The app copies the tables to `app/data/live/` on first start; edits made on the website are saved there.

---

## How the ML model was chosen

The choice is a documented experiment (`python -m app.ml.model_selection`, about 10 minutes; results in `app/ml/model_selection.json`, `docs/09_model_selection.md` and the **Model selection** page). The unit is one course attempt; all of a student's rows stay on the same side of every split (no leakage).

1. **10 classifiers** scored with the same 5-fold cross-validation repeated 3 times, grouped by student.
2. **Nested cross-validation** to tune the best ones, so tuning cannot inflate the score.
3. **Corrected resampled t-tests** (Nadeau and Bengio), then a **parsimony rule** (simplest model within one standard error of the best).
4. Inputs: prior CGPA and credits, credit hours, load that semester, level, repeat attempt, course and major, plus attendance in the "during the semester" scenario.

| Question | Selected model | Accuracy | ROC-AUC | Failures caught | "Always say passes" |
|---|---|---|---|---|---|
| Fail the course? At registration | Logistic regression | **89.8%** | 0.886 | 40% | 86.2% |
| During the semester (with attendance) | Logistic regression | **90.0%** | 0.909 | 43% | 86.2% |
| Mark (regression) | Ridge regression | typical error 7.6 / 7.2 marks; within 10 marks for 83% / 85% | | | |

**Honest reading.**
- The 85% accuracy target is met on this data, but guessing "passes" for everyone already scores 86%, because only 14% of attempts fail. Accuracy flatters the model; ROC-AUC (0.89-0.91) is the fairer measure.
- Only about 40% of failures are flagged at the default 50% alert level. Lowering it to 30% catches roughly 56-64% with more false alarms (threshold table on the study page).
- Prior CGPA is the strongest input; attendance is next once the semester has started.
- All of this describes the **simulator**. Retrain on real anonymised transcripts before relying on any number.

---

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py                   # first start creates a local login:  admin / admin123
python -m pytest tests          # optional: 30 tests
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
- Anyone on the internet can see the dashboard, analytics, semester planner and model-selection pages (aggregates only, no names). Student lists, reports, recommendations and admin pages need a login. Before loading **real UTAS records**, keep the site private (login on every page) and use a database.

Other hosts (Railway, Fly.io, a VPS) work too: set `FLASK_ENV=production`, `SECRET_KEY` and `ADMIN_PASSWORD`, and start with
`gunicorn run:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.

---

## Project structure

```
├── run.py, run.sh, run.bat, config.py, render.yaml, requirements.txt
├── tests/test_app.py             30 automated tests
├── app/
│   ├── grading.py                grade points, mark bands, standing, load limits (edit here)
│   ├── auth.py, security.py, audit.py   roles, login, CSRF, audit log
│   ├── routes/                   dashboard, students, analytics, predictions, recommendations, admin
│   ├── ml/                       preprocessing, registry (models), model_selection (study), predictor, bayes
│   ├── data/                     sample/ CSVs, generate_students.py, data_loader.py, stats.py, analytics_calc.py (semester planner)
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

MIT. The included data is simulated.
