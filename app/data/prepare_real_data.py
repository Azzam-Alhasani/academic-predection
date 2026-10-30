"""
Build the app's student dataset from the real UCI "Student Performance" data.

Source: P. Cortez and A. Silva, "Using Data Mining to Predict Secondary School
Student Performance", 2008. UCI Machine Learning Repository, licence CC BY 4.0.
Raw files (semicolon separated) live in app/data/raw/:
  - student-mat.csv  (395 students, Mathematics)
  - student-por.csv  (649 students, Portuguese language)

Run:  python app/data/prepare_real_data.py
Writes app/data/sample_students.csv (the pristine copy that the app copies to
students.csv on first start).
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
OUT_FILE = os.path.join(HERE, "sample_students.csv")

PASS_MARK = 10  # final grade G3 is on a 0-20 scale; 10 or more is a pass

COLUMNS = [
    "student_id", "name", "subject", "school", "gender", "age", "address",
    "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3", "pass_fail",
]


def _load(filename: str, subject: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, filename), sep=";", quotechar='"')
    df["subject"] = subject
    for col in ("G1", "G2", "G3"):
        df[col] = pd.to_numeric(df[col].astype(str).str.strip('"'), errors="coerce")
    return df


def build() -> pd.DataFrame:
    df = pd.concat(
        [_load("student-mat.csv", "Math"), _load("student-por.csv", "Portuguese")],
        ignore_index=True,
    )
    df = df.dropna(subset=["G1", "G2", "G3"]).reset_index(drop=True)
    for col in ("G1", "G2", "G3"):
        df[col] = df[col].astype(int)

    df["gender"] = df["sex"].map({"F": "Female", "M": "Male"})
    df["student_id"] = range(1, len(df) + 1)
    df["name"] = df["student_id"].map(lambda i: f"Student {i:04d}")
    df["pass_fail"] = (df["G3"] >= PASS_MARK).astype(int)
    return df[COLUMNS]


if __name__ == "__main__":
    out = build()
    out.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(out)} real student records -> {OUT_FILE}")
    print(out.groupby("subject").size().to_string())
    print(f"Pass rate: {out['pass_fail'].mean():.1%}")
