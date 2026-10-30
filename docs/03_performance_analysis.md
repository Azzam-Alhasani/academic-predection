# 03 — Performance Analysis

## Overview

The Analytics section of APMS provides class-wide performance insights using interactive Chart.js visualizations, all computed from the live student dataset using Pandas aggregations.

---

## Dashboard (`/`)

### KPI Cards

Six summary cards are shown at the top of every dashboard load:

| Card                | Formula / Source                                   |
|---------------------|----------------------------------------------------|
| Total Students      | `len(df)`                                          |
| Passed              | `df['pass_fail'].sum()`                            |
| Failed              | `(df['pass_fail'] == 0).sum()`                     |
| Average GPA         | `df['gpa'].mean()` rounded to 2 dp                 |
| At-Risk Students    | `df['at_risk'].sum()`                              |
| Average Attendance  | `df['attendance_pct'].mean()` rounded to 1 dp      |

### Overall Pass Rate

A colour-coded progress bar:
- **Green** — ≥ 70%
- **Yellow** — 50–69%
- **Red** — < 50%

### Grade Distribution (Bar Chart)

Counts of students in each grade category (A, B, C, D, F) using `pd.value_counts()` on the `grade_letter` derived column.

### Subject Averages (Radar Chart)

A radar showing the class mean score for each of the five subjects, computed with `df[subject_cols].mean()`. Useful for identifying which subjects the class struggles with most.

### Study Hours vs Score Trend (Line Chart)

Students are binned by `study_hours_per_week` into 5 equal-width buckets using `pd.cut()`. The mean `avg_score` is computed per bucket, revealing the correlation between study time and performance.

---

## Analytics Page (`/analytics`)

### Top 5 Performers

`df.nlargest(5, 'avg_score')` — shown with gold/silver/bronze medals.

### Bottom 5 Performers (Needs Attention)

`df.nsmallest(5, 'avg_score')` — helps teachers identify students needing intervention.

### GPA Distribution Histogram

NumPy's `np.histogram()` divides the GPA range (0–4) into 8 equal bins. Shows the distribution shape — whether grades cluster at the top, bottom, or spread normally.

### Subject Pass Rates (%)

For each subject, calculated as: `(df[subject] >= 50).mean() * 100`

Visualised as a bar chart with colour coding:
- Green — ≥ 70%
- Yellow — 50–69%
- Red — < 50%

### Subject Performance by Gender (Grouped Bar)

`df.groupby('gender')[subject_cols].mean()` — reveals subject-level performance differences by gender. Supports comparing class dynamics.

### Attendance vs Score Scatter Plot

A scatter of (attendance %, average score) pairs for up to 50 students. A positive correlation confirms attendance drives performance in the dataset.

---

## At-Risk Detection

The `at_risk` flag is automatically set to `1` if either condition is true:

```
avg_score < 50  OR  attendance_pct < 60
```

This appears as:
- A **red warning badge** on the student list table
- A **yellow "At-Risk" badge** on the student profile card
- Counted in the **KPI card** on the dashboard

---

## Grade Letter Scale

| Letter | Score Range  |
|--------|-------------|
| A      | 90–100      |
| B      | 80–89       |
| C      | 70–79       |
| D      | 60–69       |
| F      | 0–59        |

---

## GPA Conversion

```
GPA = (avg_score / 100) × 4.0
```

This maps the 0–100 score range to the standard 4.0 US GPA scale.

---

*Previous: [02 — Data Collection & Management](02_data_collection_management.md) | Next: [04 — ML Prediction Models](04_ml_prediction_models.md)*
