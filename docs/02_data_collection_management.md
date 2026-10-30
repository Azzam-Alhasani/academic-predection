# 02 — Data Collection & Management

## Overview

APMS uses a CSV file as its persistent data store. This approach requires zero database setup, works on any operating system, and can be backed up by simply copying the file.

---

## Data Schema

The student data is stored in `app/data/students.csv` (created on first run from `sample_students.csv`).

| Column                  | Type    | Range / Values      | Description                                       |
|-------------------------|---------|---------------------|---------------------------------------------------|
| `student_id`            | int     | 1–N (auto)          | Unique identifier, auto-incremented               |
| `name`                  | string  | —                   | Full name of student                              |
| `gender`                | string  | Male / Female       | Student gender                                    |
| `age`                   | int     | 10–30               | Student age in years                              |
| `attendance_pct`        | float   | 0.0–100.0           | Percentage of classes attended                    |
| `math`                  | int     | 0–100               | Mathematics score                                 |
| `science`               | int     | 0–100               | Science score                                     |
| `english`               | int     | 0–100               | English score                                     |
| `history`               | int     | 0–100               | History score                                     |
| `cs`                    | int     | 0–100               | Computer Science score                            |
| `assignments_submitted` | int     | 0–12                | Number of assignments submitted out of 12         |
| `study_hours_per_week`  | float   | 0.0–20.0            | Average study hours per week                      |
| `previous_gpa`          | float   | 0.00–4.00           | GPA from the previous academic period             |
| `pass_fail`             | int     | 0 or 1              | 0 = Fail, 1 = Pass (derived: avg_score ≥ 50)      |

### Derived Columns (computed at runtime, not stored in CSV)

| Column         | Formula                                            |
|----------------|----------------------------------------------------|
| `avg_score`    | Mean of math, science, english, history, cs        |
| `gpa`          | `avg_score / 100 × 4.0`                            |
| `grade_letter` | A (≥90), B (≥80), C (≥70), D (≥60), F (<60)       |
| `at_risk`      | 1 if avg_score < 50 OR attendance_pct < 60         |

---

## Sample Data Generation

A script generates 100 realistic synthetic students using NumPy's random distributions, with gender-appropriate Arabic names from curated name lists for male and female students:

```bash
python app/data/generate_sample_data.py
```

The script uses:
- **Beta distribution** to model previous GPA (skewed toward higher values, realistic academic distribution)
- **Normal distribution** with GPA-scaled mean to produce correlated subject scores
- **Clipping** to enforce valid score ranges
- A fixed random seed (`42`) for reproducibility

---

## Data Flow

```
1. App starts
       │
       ▼
2. config.DATA_FILE does not exist?
       │ YES                        NO
       ▼                             ▼
3. Copy sample_students.csv    Load students.csv
   → students.csv
       │
       ▼
4. data_loader.load_students()
   → reads CSV → computes derived cols → returns DataFrame
       │
       ▼
5. routes use DataFrame for display / ML training
       │
       ▼
6. On add/edit/delete: drop derived cols → write CSV
```

---

## Managing Students via the UI

### Add a Student
1. Go to **Students** page (`/students`)
2. Click **Add Student** button — a modal form appears
3. Fill in all required fields (name, gender, age, scores, etc.)
4. Click **Add Student** — the record is saved to CSV instantly

### Edit a Student
1. Click a student's name or the eye icon to open their profile
2. Click the **Edit** button — a pre-filled modal appears
3. Modify any fields and click **Save Changes**

### Delete a Student
1. On the students list, click the trash icon on any row
2. A confirmation modal appears — click **Delete**

---

## Importing Your Own Data

To replace the demo data with real data:

1. Prepare a CSV file matching the schema above (required columns only — derived columns are optional)
2. Replace `app/data/students.csv` (or `sample_students.csv` if starting fresh)
3. Restart the application — data will be loaded automatically

**Minimum required columns:** `student_id`, `name`, `gender`, `age`, `attendance_pct`, `math`, `science`, `english`, `history`, `cs`, `assignments_submitted`, `study_hours_per_week`, `previous_gpa`, `pass_fail`

---

*Previous: [01 — Project Overview](01_project_overview.md) | Next: [03 — Performance Analysis](03_performance_analysis.md)*
