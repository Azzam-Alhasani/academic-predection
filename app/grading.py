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
  * The student data comes from a university that grades 0-20, where 10-14 is a normal
    result. Converted to percentages that is 50-70% (D to C+), so a typical graduate has a
    CGPA of about 1.75. Standing thresholds are therefore scaled to the converted data
    (92% of graduates are above the "good standing" line, 54% of dropouts are below the
    "warning" line). With real UTAS marks, use the handbook values (commonly CGPA 2.0).

How the numbers are computed
  Semester GPA   = GP(avg mark of approved courses) x approved / enrolled
                   (a failed course counts as 0 grade points)
  CGPA           = (GP1 x approved1 + GP2 x approved2) / (enrolled1 + enrolled2)
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

# Academic standing from CGPA (scaled to the converted data - see the note above).
GOOD_STANDING_CGPA = 1.0
WARNING_CGPA = 0.5
MIN_COMPLETION_RATE = 0.75      # share of enrolled courses a student should pass

MARK_FORMULA = "Percentage mark = semester average grade (0-20) x 5"


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


def semester_gpa(mark, approved, enrolled):
    """Semester GPA over all enrolled courses; None when nothing was enrolled."""
    enrolled, approved = _num(enrolled), _num(approved)
    if enrolled <= 0:
        return None
    return round(points_for(mark) * min(approved, enrolled) / enrolled, 2)


def cgpa(s1_mark, s1_approved, s1_enrolled, s2_mark=None, s2_approved=0, s2_enrolled=0):
    """Cumulative GPA over the semesters on record; None when nothing was enrolled."""
    total_enrolled = _num(s1_enrolled) + _num(s2_enrolled)
    if total_enrolled <= 0:
        return None
    quality = (points_for(s1_mark) * min(_num(s1_approved), _num(s1_enrolled))
               + points_for(s2_mark) * min(_num(s2_approved), _num(s2_enrolled)))
    return round(quality / total_enrolled, 2)


def standing(cgpa_value) -> str:
    if cgpa_value is None:
        return "No courses yet"
    if cgpa_value >= GOOD_STANDING_CGPA:
        return "Good standing"
    if cgpa_value >= WARNING_CGPA:
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


def required_semester_gpa(current_cgpa, enrolled_so_far, target_cgpa, next_courses):
    """
    GPA needed over the next `next_courses` courses to reach `target_cgpa`:

        needed = (target x (E + n) - CGPA x E) / n        E = courses enrolled so far

    Returns (needed_gpa, reachable). Not reachable when it is above 4.0.
    """
    n = max(1, int(next_courses))
    e = _num(enrolled_so_far)
    needed = (target_cgpa * (e + n) - (current_cgpa or 0.0) * e) / n
    return round(max(needed, 0.0), 2), needed <= MAX_POINTS


def courses_to_pass(enrolled_so_far, approved_so_far, next_courses, target_rate=MIN_COMPLETION_RATE):
    """
    Courses to pass next semester so that overall completion reaches `target_rate`:

        k = ceil(rate x (E + n) - A)         E enrolled so far, A approved so far, n next semester
    """
    n = max(1, int(next_courses))
    needed = math.ceil(target_rate * (_num(enrolled_so_far) + n) - _num(approved_so_far))
    return max(0, min(n, needed)), needed <= n


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
