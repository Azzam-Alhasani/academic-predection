"""
Script to generate sample_students.csv with 100 realistic student records.
Run once: python app/data/generate_sample_data.py
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

MALE_FIRST_NAMES = [
    "Ahmed", "Mohamed", "Ali", "Omar", "Hassan", "Ibrahim", "Khalid",
    "Yousef", "Samir", "Tariq", "Walid", "Faisal", "Nasser", "Ziad",
    "Bilal", "Rami", "Karim", "Hani", "Sami", "Adel", "Wael", "Majid",
    "Saleh", "Fahad", "Abdullah", "Marwan", "Nabil", "Hamza", "Anas",
    "Bassem", "Tarek", "Amr", "Mahmoud", "Khaled", "Osama", "Yasser",
    "Raed", "Ayman", "Fawzi", "Jihad",
]

FEMALE_FIRST_NAMES = [
    "Fatima", "Layla", "Amira", "Nour", "Sara", "Rania", "Hana", "Dina",
    "Mona", "Aya", "Salma", "Lina", "Rana", "Heba", "Noura", "Yasmin",
    "Reem", "Sana", "Dalal", "Ghada", "Wafa", "Nada", "Ruba", "Aisha",
    "Manar", "Duaa", "Iman", "Suha", "Lubna", "Abeer", "Amal", "Basma",
    "Dalia", "Enas", "Hadeel", "Inas", "Jumana", "Khadija", "Maram", "Nawal",
]

LAST_NAMES = [
    "Al-Hassan", "Al-Rashid", "Al-Mansouri", "Al-Farsi", "Al-Zahrani",
    "Al-Siddiqui", "Al-Amin", "Al-Masri", "Al-Shammari", "Al-Otaibi",
    "Al-Harbi", "Al-Ghamdi", "Al-Qahtani", "Al-Dosari", "Al-Balushi",
    "Al-Tamimi", "Al-Nasser", "Al-Khalidi", "Al-Amiri", "Qureshi",
]


def generate_students(n=500):
    records = []
    for i in range(1, n + 1):
        gender = random.choice(["Male", "Female"])
        first = random.choice(MALE_FIRST_NAMES if gender == "Male" else FEMALE_FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        age = random.randint(15, 22)

        # Previous GPA: normal distribution centred at 2.5 → perf spread across all grade bands
        prev_gpa = round(float(np.clip(np.random.normal(2.5, 0.75), 0.3, 4.0)), 2)
        perf_factor = prev_gpa / 4.0  # 0–1

        attendance = round(float(np.clip(np.random.normal(60 + perf_factor * 35, 10), 20, 100)), 1)
        study_hours = round(float(np.clip(np.random.normal(2 + perf_factor * 14, 3), 0.5, 20)), 1)
        assignments = random.randint(max(0, int(perf_factor * 6)), 12)

        # Score range: low performer (~40) → high performer (~95), std=12 per subject
        def subject_score():
            base = 40 + perf_factor * 55 + np.random.normal(0, 12)
            return int(np.clip(base, 10, 100))

        math = subject_score()
        science = subject_score()
        english = subject_score()
        history = subject_score()
        cs = subject_score()

        avg = (math + science + english + history + cs) / 5
        pass_fail = 1 if avg >= 50 else 0

        # Previous term average: current average plus a small random change
        # (some students improve, some decline). This gives a realistic baseline
        # for computing "improvement / decline" without extra CSV tables.
        previous_term_avg = round(float(np.clip(avg + np.random.normal(0, 6), 10, 100)), 2)

        records.append({
            "student_id": i,
            "name": name,
            "gender": gender,
            "age": age,
            "attendance_pct": attendance,
            "math": math,
            "science": science,
            "english": english,
            "history": history,
            "cs": cs,
            "assignments_submitted": assignments,
            "study_hours_per_week": study_hours,
            "previous_gpa": prev_gpa,
            "previous_term_avg": previous_term_avg,
            "pass_fail": pass_fail,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sample_students.csv")
    df = generate_students(500)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} student records → {out_path}")
