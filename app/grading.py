"""
Grading rules (UTAS-style) in one place, so the whole system uses the same scale.

>>> EDIT THIS FILE to match the official UTAS student handbook. <<<

What is sourced and what is assumed
  * Percentage bands, letters and the 50% pass mark follow the grading scale commonly
    published for Omani higher education (A+ 90-100, A 85-89, B+ 80-84, B 75-79,
    C+ 70-74, C 65-69, D+ 60-64, D 50-59, F below 50).
  * ASSUMED (not found in a UTAS document): the grade points on the 4.0 scale
    (the plus grades sit half a point above the letter) and the academic-standing
    thresholds below. Change GRADE_SCALE / the thresholds if the handbook differs.
  * The data's marks are low (average about 56%, mostly D to C), so standing thresholds are
    scaled to it: in the data, students with GPA 1.0 or more almost never fail (about 1%),
    those between 0.5 and 1.0 fail about 3 times in 10, and those under 0.5 about 8 times in 10.
    With real UTAS marks, use the handbook values (commonly 2.0 for good standing).

How the numbers are computed (one record = one student in one course)
  Term GPA    = grade points of the term mark                       (e.g. 62% -> D+ -> 1.5)
  GPA         = average of the term GPAs on record
  Average mark= average of the term marks on record
  Goal planner: the final mark needed so the three marks average to a target (equal weights assumed)
"""

import math

# (minimum mark, letter, grade points, description) - highest band first.
GRADE_SCALE = [
    (90, "A+", 4.0, "Excellent"),
    (85, "A", 4.0, "Very good"),
    (80, "B+", 3.5, "Good"),
    (75, "B", 3.0, "Above average"),
    (70, "C+", 2.5, "Satisfactory"),
    (65, "C", 2.0, "Pass"),
    (60, "D+", 1.5, "Marginal pass"),
    (50, "D", 1.0, "Weak pass"),
    (0, "F", 0.0, "Fail"),
]

PASS_MARK = 50.0
MAX_POINTS = 4.0

# Academic standing from GPA (scaled to the data - see the note above).
GOOD_STANDING_GPA = 1.0
WARNING_GPA = 0.5
ABSENCE_BANDS = [(0, 0, "No absences"), (1, 5, "1-5 absences"), (6, 10, "6-10 absences"), (11, 10**6, "11+ absences")]

MARK_FORMULA = "Percentage mark = grade (0-20) x 5"


def grade_for(mark) -> dict:
    """Letter, grade points and description for a 0-100 mark."""
    mark = _clean(mark)
    for minimum, letter, points, description in GRADE_SCALE:
        if mark >= minimum:
            return {"letter": letter, "points": points, "description": description,
                    "passed": mark >= PASS_MARK}
    return {"letter": "F", "points": 0.0, "description": "Fail", "passed": False}


def letter_for(mark) -> str:
    return grade_for(mark)["letter"]


def points_for(mark) -> float:
    return grade_for(mark)["points"]


def term_gpa(mark):
    """Grade points of one term mark."""
    return points_for(mark)


def gpa(*marks):
    """Average grade points of the term marks given (None values skipped); None when there are none."""
    values = [points_for(m) for m in marks if m is not None and not (isinstance(m, float) and math.isnan(m))]
    return round(sum(values) / len(values), 2) if values else None


def average_mark(*marks):
    values = [_clean(m) for m in marks if m is not None and not (isinstance(m, float) and math.isnan(m))]
    return round(sum(values) / len(values), 1) if values else None


def absence_band(absences) -> str:
    n = int(_num(absences))
    for low, high, label in ABSENCE_BANDS:
        if low <= n <= high:
            return label
    return ABSENCE_BANDS[-1][2]


def standing(gpa_value) -> str:
    if gpa_value is None:
        return "No courses yet"
    if gpa_value >= GOOD_STANDING_GPA:
        return "Good standing"
    if gpa_value >= WARNING_GPA:
        return "Academic warning"
    return "Academic probation"


def next_grade_up(mark):
    """(next letter, mark needed, marks short) or None if already at the top grade."""
    mark = _clean(mark)
    higher = [band for band in GRADE_SCALE if band[0] > mark]
    if not higher:
        return None
    minimum, letter, _, _ = higher[-1]          # the closest band above
    return {"letter": letter, "mark": minimum, "gap": round(minimum - mark, 1)}


def mark_for_points(target_points):
    """Lowest mark whose grade points are >= target_points (None if above the scale)."""
    if target_points > MAX_POINTS:
        return None
    candidates = [minimum for minimum, _, points, _ in GRADE_SCALE if points >= target_points]
    return float(min(candidates)) if candidates else None


def required_final_mark(t1, t2, target_average):
    """
    Final mark needed so that the three marks average to `target_average` (equal weights):

        needed = 3 x target - term 1 - term 2

    Returns (needed mark, reachable). Not reachable when it is above 100.
    """
    needed = 3 * _num(target_average) - _num(t1) - _num(t2)
    return round(max(needed, 0.0), 1), needed <= 100


def _clean(value) -> float:
    number = _num(value)
    return max(0.0, min(100.0, number))


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def scale_table() -> list:
    """The scale as dicts, for templates."""
    rows = []
    upper = 100
    for minimum, letter, points, description in GRADE_SCALE:
        rows.append({"letter": letter, "range": f"{minimum}-{upper}" if minimum else f"0-{upper}",
                     "points": points, "description": description})
        upper = minimum - 1
    return rows
