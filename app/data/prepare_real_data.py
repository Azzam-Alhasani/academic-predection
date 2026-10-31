"""
Build the app's student table from the real UCI "Student Performance" data (Cortez & Silva, 2008).

Run:  python app/data/prepare_real_data.py
Writes app/data/sample_students.csv, the pristine copy that the app copies to students.csv.

Only the factors that matter for results are kept (see raw/README.md):
    course, term 1 mark, term 2 mark, absences, final mark
Marks are converted from 0-20 to percentages (x 5) so they read like a UTAS transcript.
`person_id` is an internal link between a student's two courses (never shown); it keeps both
rows on the same side of cross-validation splits.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT_FILE = os.path.join(HERE, "sample_students.csv")

COURSES = {"student-mat.csv": "Mathematics", "student-por.csv": "Portuguese"}
LINK_COLUMNS = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
                "Mjob", "Fjob", "reason", "nursery", "internet"]
COLUMNS = ["student_id", "name", "person_id", "course", "t1_mark", "t2_mark", "absences", "final_mark"]


def build() -> pd.DataFrame:
    frames = []
    for filename, course in COURSES.items():
        raw = pd.read_csv(os.path.join(RAW, filename), sep=";", quotechar='"', skipinitialspace=True)
        raw["course"] = course
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    for col in ("G1", "G2", "G3", "absences"):
        raw[col] = pd.to_numeric(raw[col])
    person = raw[LINK_COLUMNS].astype(str).agg("|".join, axis=1).factorize()[0] + 1
    out = pd.DataFrame({
        "person_id": person,
        "course": raw["course"],
        "t1_mark": (raw["G1"] * 5).round(1),
        "t2_mark": (raw["G2"] * 5).round(1),
        "absences": raw["absences"].astype(int),
        "final_mark": (raw["G3"] * 5).round(1),
    })
    out.insert(0, "student_id", range(1, len(out) + 1))
    out.insert(1, "name", out["student_id"].map(lambda i: f"Student {i:04d}"))
    return out[COLUMNS]


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} real student records ({df.person_id.nunique()} different students) -> {OUT_FILE}")
    print(df.groupby("course").size().to_string())
    print(f"Final mark below 50 (fail): {(df.final_mark < 50).mean():.1%}")
