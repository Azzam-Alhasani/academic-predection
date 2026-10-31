"""
Build the app's student table from the real UCI "Predict Students' Dropout and
Academic Success" data (4,424 university students, see raw/README.md).

Run:  python app/data/prepare_real_data.py
Writes app/data/sample_students.csv, the pristine copy that the app copies to
students.csv on first start.

Conversions (so the numbers read like a UTAS transcript):
  * semester average grade 0-20  ->  percentage mark 0-100   (x 5)
  * admission / previous-qualification grade 0-200  ->  0-100 (/ 2)
  * "curricular units" are called courses
  * 0/1 flags become no/yes; Gender 1 = Male, 0 = Female (as documented by the authors)
Letters, GPA, CGPA and academic standing are computed from the marks by app/grading.py.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(HERE, "raw", "uci_dropout_success.csv")
OUT_FILE = os.path.join(HERE, "sample_students.csv")

# Course codes as documented in the dataset paper.
PROGRAMMES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)",
}

COLUMNS = [
    "student_id", "name", "gender", "age", "programme", "attendance", "admission_mark",
    "prev_mark", "scholarship", "tuition_paid", "debtor",
    "s1_enrolled", "s1_approved", "s1_mark", "s2_enrolled", "s2_approved", "s2_mark", "outcome",
]


def _yes_no(series: pd.Series) -> pd.Series:
    return series.map({1: "yes", 0: "no"})


def build() -> pd.DataFrame:
    raw = pd.read_csv(RAW_FILE, sep=";")
    out = pd.DataFrame({
        "gender": raw["Gender"].map({1: "Male", 0: "Female"}),
        "age": raw["Age at enrollment"],
        "programme": raw["Course"].map(PROGRAMMES),
        "attendance": raw["Daytime/evening attendance"].map({1: "Daytime", 0: "Evening"}),
        "admission_mark": (raw["Admission grade"] / 2).round(1),
        "prev_mark": (raw["Previous qualification (grade)"] / 2).round(1),
        "scholarship": _yes_no(raw["Scholarship holder"]),
        "tuition_paid": _yes_no(raw["Tuition fees up to date"]),
        "debtor": _yes_no(raw["Debtor"]),
        "s1_enrolled": raw["Curricular units 1st sem (enrolled)"],
        "s1_approved": raw["Curricular units 1st sem (approved)"],
        "s1_mark": (raw["Curricular units 1st sem (grade)"] * 5).round(1),
        "s2_enrolled": raw["Curricular units 2nd sem (enrolled)"],
        "s2_approved": raw["Curricular units 2nd sem (approved)"],
        "s2_mark": (raw["Curricular units 2nd sem (grade)"] * 5).round(1),
        "outcome": raw["Target"],
    })
    if out["programme"].isna().any():
        raise ValueError("Unknown course code in the raw file")
    out.insert(0, "student_id", range(1, len(out) + 1))
    out.insert(1, "name", out["student_id"].map(lambda i: f"Student {i:04d}"))
    return out[COLUMNS]


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} real student records -> {OUT_FILE}")
    print(df["outcome"].value_counts().to_string())
    print(f"Dropout rate: {(df['outcome'] == 'Dropout').mean():.1%}")
