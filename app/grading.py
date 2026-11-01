"""
Grading rules (UTAS-style) in one place, so the whole system uses the same scale.

>>> EDIT THIS FILE to match the official UTAS handbook. <<<

What is sourced and what is assumed
  * SOURCED (from a UTAS transcript key): letter -> grade points
        A 4.0, A- 3.7, B+ 3.3, B 3.0, B- 2.7, C+ 2.3, C 2.0, C- 1.7, D+ 1.3, D 1.0, F 0.0
    and the rules visible on that transcript: a withdrawn course (W) has no grade points and is not
    counted as attempted; a repeated course is marked as a repeat; C- and D carry result "F"
    (fail), C and above carry result "P" (pass).
  * ASSUMED (no official table available): the percentage band for each letter below, hence the
    pass mark of 60 (the lower edge of C); the academic-standing thresholds; the credit-load limits.

How the numbers are computed
  Quality points = credit hours x grade points             (W earns nothing and is not attempted)
  SGPA  = quality points of the semester / credits attempted in the semester
  CGPA  = quality points of all attempts / credits attempted in all attempts
  LCGPA = like CGPA but a repeated course counts only through its latest attempt
  Credits earned = credit hours of passed courses
"""

import math

# (minimum mark, letter, grade points, description) - highest band first.
GRADE_SCALE = [
    (90, "A", 4.0, "Excellent"),
    (85, "A-", 3.7, "Very good"),
    (80, "B+", 3.3, "Good"),
    (75, "B", 3.0, "Good"),
    (70, "B-", 2.7, "Above average"),
    (65, "C+", 2.3, "Satisfactory"),
    (60, "C", 2.0, "Pass"),
    (55, "C-", 1.7, "Fail"),
    (50, "D+", 1.3, "Fail"),
    (45, "D", 1.0, "Fail"),
    (0, "F", 0.0, "Fail"),
]
POINTS = {letter: pts for _, letter, pts, _ in GRADE_SCALE}
PASS_POINTS = 2.0                 # a course is passed with C (2.0) or better
PASS_MARK = 60.0                  # lower edge of C (assumed band)
MAX_POINTS = 4.0

# Academic standing from CGPA (assumed).
GOOD_STANDING_CGPA = 2.0
WARNING_CGPA = 1.5

# Credit load per semester (assumed): below LIGHT = light, above NORMAL_MAX = overload.
LIGHT_LOAD, NORMAL_MAX = 12, 18

ATTENDANCE_BANDS = [(90, "90-100%"), (80, "80-89%"), (70, "70-79%"), (0, "Below 70%")]
ATTENDANCE_ORDER = [b[1] for b in reversed(ATTENDANCE_BANDS)]


def grade_for(mark) -> dict:
    """Letter, grade points and description for a 0-100 mark."""
    mark = _clean(mark)
    for minimum, letter, points, description in GRADE_SCALE:
        if mark >= minimum:
            return {"letter": letter, "points": points, "description": description, "passed": points >= PASS_POINTS}
    return {"letter": "F", "points": 0.0, "description": "Fail", "passed": False}


def letter_for(mark) -> str:
    return grade_for(mark)["letter"]


def points_for(mark) -> float:
    return grade_for(mark)["points"]


def passed(mark) -> bool:
    return grade_for(mark)["passed"]


def expected_points(mean_mark, sd) -> float:
    """Expected grade points when the mark is Normal(mean, sd): sum of points x P(mark in each band)."""
    sd = max(float(sd), 0.5)

    def cdf(x):
        return 0.5 * (1 + math.erf((x - mean_mark) / (sd * math.sqrt(2))))

    total, upper = 0.0, 100.5
    for minimum, _, points, _ in GRADE_SCALE:
        lower = minimum - 0.5 if minimum > 0 else -1e9
        total += points * (cdf(upper) - cdf(lower))
        upper = lower
    return total


def load_status(credits) -> str:
    credits = _num(credits)
    return "Light Load" if credits < LIGHT_LOAD else "Normal Load" if credits <= NORMAL_MAX else "Overload"


def standing(cgpa_value) -> str:
    if cgpa_value is None or (isinstance(cgpa_value, float) and math.isnan(cgpa_value)):
        return "No courses yet"
    if cgpa_value >= GOOD_STANDING_CGPA:
        return "Good standing"
    if cgpa_value >= WARNING_CGPA:
        return "Academic warning"
    return "Academic probation"


def attendance_band(att) -> str:
    att = _num(att)
    for minimum, label in ATTENDANCE_BANDS:
        if att >= minimum:
            return label
    return ATTENDANCE_BANDS[-1][1]


def next_grade_up(mark):
    """Next letter above the mark: {letter, mark, gap}, or None at the top grade."""
    mark = _clean(mark)
    higher = [band for band in GRADE_SCALE if band[0] > mark]
    if not higher:
        return None
    minimum, letter, _, _ = higher[-1]
    return {"letter": letter, "mark": minimum, "gap": round(minimum - mark, 1)}


def required_sgpa(cgpa_value, attempted, target, next_credits):
    """
    SGPA needed over the next `next_credits` credits to reach `target` CGPA:

        needed = (target x (A + n) - CGPA x A) / n          A = credits attempted so far

    Returns (needed, reachable). Not reachable when it is above 4.0.
    """
    n = max(1.0, _num(next_credits))
    a = _num(attempted)
    needed = (target * (a + n) - _num(cgpa_value) * a) / n
    return round(max(needed, 0.0), 2), needed <= MAX_POINTS


def _clean(value) -> float:
    return max(0.0, min(100.0, _num(value)))


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def scale_table() -> list:
    """The scale as dicts, for templates."""
    rows, upper = [], 100
    for minimum, letter, points, description in GRADE_SCALE:
        rows.append({"letter": letter, "range": f"{minimum}-{upper}", "points": points, "description": description})
        upper = minimum - 1
    return rows
