"""
Generate a SIMULATED student-transcript dataset, shaped like a UTAS transcript.

>>> The student records are NOT real. <<<
No public dataset has course-by-course marks with attendance for several majors, so this script
simulates one. What it contains and how:

  courses.csv      course catalogue: code, name, major (or Common), credit hours, level
  students.csv     student_id, name, gender, major, entry_semester
  enrollments.csv  one row per student per course attempt: semester, course, mark, attendance %, status
                   (Completed / Withdrawn / In progress). Marks are blank for W and in-progress rows.

Simulation rules (assumptions, written down so they can be challenged):
  mark = 75 + 8 x ability + course difficulty + attendance effect - overload penalty + noise(sd 6)
  * ability ~ N(0,1) per student; course difficulty is a fixed offset per course (Calculus/Physics hard,
    project and studio courses easier); the offsets live ONLY in this file - the app measures difficulty
    from the marks, the way it would on real data.
  * attendance effect: -0.28 marks per attendance point below 88%, +0.08 per point above; attendance
    is correlated with ability (r about 0.3).
  * a failed course (grade below C, i.e. mark < 60) is usually repeated (at most two repeats per
    semester, the rest wait their turn); a withdrawn course sometimes is; students with low expected
    marks withdraw more often; more than 18 credits in one semester costs 0.5 marks per extra credit.
  * gender has NO effect on marks (it is included only for reporting).
To use real data, replace the three CSVs (same columns) with anonymised transcripts.

Run:  python app/data/generate_students.py
"""

import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample")
SEED = 2026

# code|name|credit hours|difficulty offset (marks; negative = harder). Offsets are used only here.
COMMON = """ENG101|English for Academic Purposes I|3|-1
ENG102|Academic Writing II|3|-1
ENG201|Public Speaking|3|1
ETH201|Working Ethics|2|3
ENT201|Entrepreneurship and Innovation|3|2
ARB101|Arabic Language Skills|3|1
MATH101|Calculus I|3|-5
MATH102|Calculus II|3|-7
MATH201|Engineering Mathematics|3|-7
STAT201|Statistics|3|-3
PHYS101|Physics I|3|-6
PHYS102|Physics II|4|-7
CHEM101|General Chemistry|4|-4"""

MAJORS = {
    "Engineering": [
        ["ENG101", "MATH101", "PHYS101", "CHEM101", "EGR101|Engineering Graphics|3|1"],
        ["ENG102", "MATH102", "PHYS102", "EGR102|Programming for Engineers|3|-2", "EGR103|Engineering Workshop|3|3"],
        ["MATH201", "EGR201|Circuits I|3|-5", "EGR202|Digital Logic Design|3|-2", "EGR203|Materials Science|3|-3", "STAT201"],
        ["EGR204|Electronics|3|-4", "EGR205|Thermodynamics|3|-5", "EGR206|Microprocessors|3|-3", "ENT201", "ETH201"],
        ["EGR301|Control Systems|3|-6", "EGR302|Signals and Systems|3|-6", "EGR303|Computer Networks|3|-2", "EGR304|Project Management|3|2", "ENG201"],
        ["EGR401|Embedded Systems|3|-2", "EGR402|Robotics and Automation|3|-1", "EGR403|Capstone Project I|3|3", "EGR404|Research Methodology|3|0", "ARB101"],
    ],
    "Business": [
        ["ENG101", "BUS101|Principles of Management|3|1", "BUS102|Introduction to Accounting|3|-2", "ECO101|Microeconomics|3|-2", "BUS103|Business Mathematics|3|-3"],
        ["ENG102", "BUS104|Financial Accounting|3|-3", "BUS105|Marketing Principles|3|2", "ECO102|Macroeconomics|3|-3", "BUS106|Business Communication|3|2"],
        ["STAT201", "BUS201|Managerial Accounting|3|-4", "BUS202|Organizational Behaviour|3|2", "BUS203|Business Law|3|0", "ENT201"],
        ["BUS204|Corporate Finance|3|-6", "BUS205|Operations Management|3|-2", "BUS206|Human Resource Management|3|2", "BUS207|Business Analytics|3|-3", "ETH201"],
        ["BUS301|Investment Analysis|3|-5", "BUS302|International Business|3|1", "BUS303|Strategic Management|3|0", "BUS304|Consumer Behaviour|3|2", "ENG201"],
        ["BUS401|Business Research|3|-1", "BUS402|Entrepreneurial Venture|3|2", "BUS403|Capstone Project|3|3", "BUS404|Risk Management|3|-2", "ARB101"],
    ],
    "Information Technology": [
        ["ENG101", "MATH101", "ITC101|Introduction to Programming|3|-3", "ITC102|Computer Fundamentals|3|2", "ITC103|Digital Logic|3|-2"],
        ["ENG102", "ITC104|Discrete Mathematics|3|-6", "ITC105|Object-Oriented Programming|3|-4", "ITC106|Web Development|3|2", "STAT201"],
        ["ITC201|Data Structures|3|-6", "ITC202|Database Systems|3|-3", "ITC203|Computer Networks|3|-2", "ITC204|Operating Systems|3|-4", "ENT201"],
        ["ITC205|Software Engineering|3|-1", "ITC206|Web Applications|3|1", "ITC207|Network Security|3|-3", "ITC208|Algorithms|3|-6", "ETH201"],
        ["ITC301|Machine Learning|3|-5", "ITC302|Mobile Applications|3|0", "ITC303|Cloud Computing|3|-1", "ITC304|Human-Computer Interaction|3|2", "ENG201"],
        ["ITC401|Project I|3|3", "ITC402|IT Project Management|3|2", "ITC403|Cybersecurity|3|-3", "ITC404|Research Methodology|3|0", "ARB101"],
    ],
    "Applied Sciences": [
        ["ENG101", "MATH101", "CHEM101", "BIO101|General Biology|3|-1", "PHYS101"],
        ["ENG102", "STAT201", "CHEM102|Organic Chemistry|4|-7", "BIO102|Cell Biology|3|-2", "LAB101|Laboratory Skills|2|4"],
        ["CHEM201|Analytical Chemistry|4|-5", "BIO201|Biochemistry|3|-5", "MATH102", "ENV201|Environmental Science|3|1", "ENT201"],
        ["BIO202|Microbiology|3|-2", "CHEM202|Physical Chemistry|3|-7", "LAB201|Instrumental Analysis|3|-2", "BIO203|Genetics|3|-3", "ETH201"],
        ["BIO301|Molecular Biology|3|-3", "CHEM301|Spectroscopy|3|-4", "ENV301|Water Quality|3|0", "SCI301|Research Design|3|0", "ENG201"],
        ["SCI401|Research Project I|3|3", "BIO401|Biotechnology|3|0", "CHEM401|Industrial Chemistry|3|-2", "SCI402|Quality Assurance|3|1", "ARB101"],
    ],
    "Design": [
        ["ENG101", "DES101|Design Fundamentals|3|2", "DES102|Drawing|3|2", "DES103|Colour Theory|3|3", "ART101|History of Art|3|1"],
        ["ENG102", "DES104|Digital Design Tools|3|1", "DES105|Typography|3|1", "DES106|Materials and Processes|3|0", "ARB101"],
        ["DES201|Visual Communication|3|1", "DES202|3D Modelling|3|-3", "DES203|Design Research|3|-1", "ENT201", "DES204|Photography|3|2"],
        ["DES205|Branding|3|2", "DES206|Interaction Design|3|-2", "DES207|Studio Project I|3|2", "ETH201", "DES208|Packaging Design|3|0"],
        ["DES301|Studio Project II|3|2", "DES302|UX Research|3|-2", "DES303|Motion Graphics|3|-1", "DES304|Design Management|3|1", "ENG201"],
        ["DES401|Capstone Project|3|3", "DES402|Portfolio|3|3", "DES403|Professional Practice|3|2", "DES404|Design Research Methods|3|0", "STAT201"],
    ],
}
MAJOR_WEIGHTS = {"Engineering": 0.28, "Business": 0.24, "Information Technology": 0.22, "Applied Sciences": 0.14, "Design": 0.12}

SEMESTERS = ["2023 Fall", "2024 Spring", "2024 Fall", "2025 Spring", "2025 Fall", "2026 Spring", "2026 Fall"]
CURRENT = "2026 Fall"
COHORTS = {"2023 Fall": 150, "2024 Fall": 260, "2025 Fall": 260, "2026 Fall": 180}


def _parse(spec):
    code, name, credits, diff = spec.split("|")
    return code, name, int(credits), float(diff)


def build_catalogue():
    base = {c[0]: c for c in map(_parse, COMMON.splitlines())}
    courses, diffs, plans = {}, {}, {}
    for code, (_, name, credits, diff) in base.items():
        courses[code] = {"course_code": code, "course_name": name, "major": "Common", "credit_hours": credits}
        diffs[code] = diff
    for major, semesters in MAJORS.items():
        plan = []
        for sem in semesters:
            codes = []
            for item in sem:
                if "|" in item:
                    code, name, credits, diff = _parse(item)
                    courses[code] = {"course_code": code, "course_name": name, "major": major, "credit_hours": credits}
                    diffs[code] = diff
                else:
                    code = item
                codes.append(code)
            plan.append(codes)
        plans[major] = plan
    frame = pd.DataFrame(courses.values())
    frame["level"] = frame["course_code"].str.extract(r"(\d)")[0].astype(int)
    return frame, diffs, plans


def attendance_effect(att):
    return 0.28 * (att - 88) if att < 88 else 0.08 * (att - 88)


def simulate():
    rng = np.random.default_rng(SEED)
    rnd = random.Random(SEED)
    courses, diffs, plans = build_catalogue()
    credit = dict(zip(courses.course_code, courses.credit_hours))
    majors, weights = list(MAJOR_WEIGHTS), list(MAJOR_WEIGHTS.values())

    students, rows = [], []
    sid = 0
    for entry, count in COHORTS.items():
        start = SEMESTERS.index(entry)
        for _ in range(count):
            sid += 1
            major = rnd.choices(majors, weights)[0]
            ability = rng.normal()
            students.append({"student_id": sid, "name": f"Student {sid:04d}", "gender": rnd.choice(["Male", "Female"]),
                             "major": major, "entry_semester": entry})
            att_base = 88 + 5 * ability + rng.normal(0, 6)
            queue = []                        # course codes waiting to be retaken
            for k, semester in enumerate(SEMESTERS[start:]):
                curriculum = plans[major][k] if k < len(plans[major]) else []
                retake = [c for c in queue if c not in curriculum][:2]
                queue = [c for c in queue if c not in retake and c not in curriculum]
                load = list(dict.fromkeys(curriculum + retake))
                repeats = []
                if not load:
                    continue
                total = sum(credit[c] for c in load)
                current = semester == CURRENT
                for code in load:
                    att = float(np.clip(att_base + rng.normal(0, 5), 35, 100))
                    expected = 75 + 8 * ability + diffs[code] + attendance_effect(att) - 0.5 * max(0, total - 18)
                    mark = float(np.clip(expected + rng.normal(0, 6.0), 0, 100))
                    p_withdraw = 0.008 + 0.14 / (1 + np.exp((expected - 54) / 4.5))
                    status, value = "Completed", int(round(mark))
                    if current:
                        status, value = "In progress", None
                    elif rng.random() < p_withdraw:
                        status, value = "Withdrawn", None
                    rows.append({"student_id": sid, "semester": semester, "course_code": code,
                                 "mark": value, "attendance": round(att, 1), "status": status})
                    if status == "Withdrawn" and rng.random() < 0.7:
                        repeats.append(code)
                    elif status == "Completed" and value < 60 and rng.random() < 0.9:
                        repeats.append(code)
                queue += repeats
    enrollments = pd.DataFrame(rows)
    enrollments.insert(0, "enrollment_id", range(1, len(enrollments) + 1))
    return courses, pd.DataFrame(students), enrollments


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    courses, students, enrollments = simulate()
    courses.to_csv(os.path.join(OUT, "courses.csv"), index=False)
    students.to_csv(os.path.join(OUT, "students.csv"), index=False)
    enrollments.to_csv(os.path.join(OUT, "enrollments.csv"), index=False)
    done = enrollments[enrollments.status == "Completed"]
    print(f"{len(courses)} courses, {len(students)} students, {len(enrollments)} enrollments")
    print(enrollments.status.value_counts().to_string())
    print(f"mean mark {done.mark.mean():.1f}, share below 60: {(done.mark < 60).mean():.1%}")
