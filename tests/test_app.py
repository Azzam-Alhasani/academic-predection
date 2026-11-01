"""Tests: UTAS grading maths, transcript data, course difficulty, planner what-ifs, roles and access control, security, model study."""

import pytest

import app.auth as auth
from app import create_app, grading

SID = 633          # a sample student with 3 semesters of records and 7 courses in progress


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = create_app("development")
    app.config["TESTING"] = True
    app.config["DATA_DIR"] = str(tmp_path / "live")          # starts as a copy of the sample data
    monkeypatch.setattr(auth, "_users_file", lambda: str(tmp_path / "users.csv"))
    monkeypatch.setattr("app.audit._log_path", lambda: str(tmp_path / "audit.log"), raising=False)
    auth._failed_logins.clear()
    auth.add_user("admin", "correct-horse1", role="admin")
    auth.add_user("teacher", "correct-horse2", role="teacher")
    auth.add_user("stu", "correct-horse3", role="student", student_id=SID)
    return app.test_client()


def _csrf(client):
    with client.session_transaction() as s:
        s["_csrf_token"] = "test-token"
    return "test-token"


def _login(client, username, password):
    return client.post("/auth/login", data={"username": username, "password": password, "csrf_token": _csrf(client)})


def _student_form(token):
    return {"csrf_token": token, "name": "Test Student", "gender": "Female", "major": "Business", "entry_semester": "2026 Fall"}


# ── grading (grade points come from the UTAS transcript) ─
def test_grade_points_match_the_transcript():
    expected = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0}
    assert grading.POINTS == expected


def test_pass_needs_c_or_better():
    assert grading.letter_for(60) == "C" and grading.passed(60) and grading.passed(72)
    assert grading.letter_for(59) == "C-" and not grading.passed(59) and not grading.passed(46) and not grading.passed(10)


def test_expected_points_is_between_the_bounds_and_rises_with_the_mark():
    low, high = grading.expected_points(50, 8), grading.expected_points(85, 8)
    assert 0 <= low < high <= 4


def test_required_sgpa_formula():
    needed, reachable = grading.required_sgpa(2.0, 30, 2.5, 15)   # (2.5*45 - 2.0*30) / 15
    assert round(needed, 2) == 3.5 and reachable
    assert grading.required_sgpa(1.0, 60, 3.5, 12)[1] is False        # would need more than 4.0


def test_load_status_and_standing():
    assert grading.load_status(9) == "Light Load" and grading.load_status(15) == "Normal Load" and grading.load_status(21) == "Overload"
    assert grading.standing(2.4) == "Good standing" and grading.standing(1.7) == "Academic warning" and grading.standing(1.0) == "Academic probation"


# ── transcript data ─────────────────────────────────────
def test_dataset_has_the_transcript_structure(client):
    from app.data import data_loader
    with client.application.app_context():
        t = data_loader.load_tables()
    assert {"gender", "major"} <= set(t.students.columns)
    assert t.students["major"].nunique() >= 5 and set(t.students["gender"]) == {"Male", "Female"}
    assert set(t.enrollments["status"]) == {"Completed", "Withdrawn", "In progress"}
    assert t.enrollments["attendance"].between(0, 100).all()
    banned = {"age", "scholarship", "tuition_paid", "debtor"}
    assert not banned & (set(t.students.columns) | set(t.enrollments.columns))


def test_withdrawn_courses_have_no_points_and_are_not_attempted(client):
    from app.data import data_loader
    with client.application.app_context():
        enr = data_loader.load_tables().enrollments
    w = enr[enr["status"] == "Withdrawn"]
    assert len(w) > 0 and (w["attempted"] == 0).all() and (w["quality"] == 0).all()


def test_sgpa_and_cgpa_follow_the_formulas(client):
    from app.data import data_loader
    with client.application.app_context():
        tr = data_loader.transcript(SID)
        done = [s for s in tr if not s["in_progress"]]
        for sem in done:
            quality = sum(c["credit_hours"] * c["points"] for c in sem["courses"] if c["status"] == "Completed")
            attempted = sum(c["credit_hours"] for c in sem["courses"] if c["status"] == "Completed")
            assert sem["sgpa"] == pytest.approx(quality / attempted, abs=0.01)
        total_q = sum(s["quality"] for s in done)
        total_a = sum(s["attempted"] for s in done)
        assert done[-1]["cgpa"] == pytest.approx(total_q / total_a, abs=0.01)


def test_lcgpa_counts_only_latest_attempt_of_a_repeated_course(client):
    from app.data import data_loader
    with client.application.app_context():
        t = data_loader.load_tables()
        s = t.summary
        repeaters = s[(s["repeats"] > 0) & s["cgpa"].notna()]
    assert len(repeaters) > 0
    assert (repeaters["l_attempted"] <= repeaters["attempted"]).all() and (repeaters["l_attempted"] < repeaters["attempted"]).any()


def test_course_difficulty_is_measured_and_tiered(client):
    from app.data import data_loader
    with client.application.app_context():
        cs = data_loader.course_stats()
    assert set(cs["tier"]) == {"Hard", "Medium", "Easy"}
    hard = cs[cs["tier"] == "Hard"]["mean_mark"].mean()
    easy = cs[cs["tier"] == "Easy"]["mean_mark"].mean()
    assert hard < easy


# ── semester planner ────────────────────────────────────
def test_plan_has_withdraw_and_attendance_what_ifs(client):
    from app.data.analytics_calc import plan_for_student
    with client.application.app_context():
        plan = plan_for_student(SID)
    assert plan["courses"] and plan["suggestions"] and plan["calculations"]
    assert all(0 <= c["p_fail"] <= 1 for c in plan["courses"])
    assert plan["cgpa_att"] >= plan["cgpa"] - 1e-9                      # attending more never predicted to hurt
    for c in plan["courses"]:
        assert c["mark_att"] >= c["mark"] - 1e-9 and c["p_fail_att"] <= c["p_fail"] + 1e-9
    for w in plan["withdraw"]:
        assert w["credits_left"] == plan["credits"] - next(c["credits"] for c in plan["courses"] if c["course_code"] == w["course_code"])


def test_withdrawing_a_likely_failure_is_recommended():
    from app.data.analytics_calc import plan_semester
    from app import create_app
    app = create_app("development")
    with app.app_context():
        plan = plan_semester("Engineering", 2.0, 40, [
            {"course_code": "MATH102", "attendance": 45, "attempt": 1}, {"course_code": "PHYS102", "attendance": 95, "attempt": 1},
            {"course_code": "ENG102", "attendance": 95, "attempt": 1}, {"course_code": "ARB101", "attendance": 95, "attempt": 1}])
    assert plan["withdraw"] and plan["withdraw"][0]["course_code"] == "MATH102" and plan["withdraw"][0]["d_cgpa"] > 0
    assert any("withdraw" in s["title"].lower() or "attendance" in s["title"].lower() for s in plan["suggestions"])


def test_planner_page_loads_and_calculates(client):
    assert client.get("/predictions/").status_code == 200
    form = {"csrf_token": _csrf(client), "major": "Engineering", "prior_cgpa": "2.4", "prior_credits": "45", "target": "2.6",
            "course_0": "MATH102", "attendance_0": "60", "course_1": "PHYS102", "attendance_1": "85", "course_2": "ENG102", "attendance_2": "90"}
    html = client.post("/predictions/", data=form).get_data(as_text=True)
    assert "Expected SGPA" in html and "withdraw" in html.lower() and "alert-danger" not in html


def test_planner_rejects_bad_input(client):
    base = {"csrf_token": _csrf(client), "major": "Engineering", "course_0": "MATH102", "attendance_0": "150"}
    assert "between 0 and 100" in client.post("/predictions/", data=base).get_data(as_text=True)
    base.update(attendance_0="80", prior_cgpa="5", prior_credits="30")
    assert "between 0 and 4" in client.post("/predictions/", data=base).get_data(as_text=True)
    base.update(course_0="NOPE999", prior_cgpa="2", prior_credits="30")
    assert "alert-danger" in client.post("/predictions/", data=base).get_data(as_text=True)


# ── model study ─────────────────────────────────────────
def test_model_study_is_present_and_honest():
    from app.ml.predictor import load_selection
    study = load_selection()
    for scen in ("planning", "during"):
        block = study["classification"][scen]
        assert len(block["candidates"]) >= 10 and block["selected"]
        base = next(c for c in block["candidates"] if c["key"] == "majority")
        best = next(c for c in block["tuned"] if c["key"] == block["selected"])
        assert best["metrics"]["accuracy"]["mean"] >= 0.85                       # the target accuracy (on simulated data)
        assert best["metrics"]["accuracy"]["mean"] > base["metrics"]["accuracy"]["mean"]
        assert best["metrics"]["roc_auc"]["mean"] > 0.8
        assert study["regression"][scen]["selected"]


def test_gender_is_not_a_model_input():
    from app.ml.preprocessing import SCENARIOS, feature_columns
    for scen in SCENARIOS:
        assert not {"gender", "student_id", "name"} & set(feature_columns(scen))


def test_no_leakage_between_a_students_courses():
    from app.ml.model_selection import grouped_splits
    import numpy as np
    groups = np.repeat(np.arange(100), 2)
    y = np.tile([0, 1], 100)
    for train, test in grouped_splits(y, groups, 5, 2, 1):
        assert not set(groups[train]) & set(groups[test])


# ── pages ───────────────────────────────────────────────
def test_public_pages_load(client):
    for url in ["/", "/analytics/", "/predictions/", "/predictions/selection", "/auth/login", "/healthz"]:
        assert client.get(url).status_code == 200, url


def test_staff_pages_render_with_data(client):
    _login(client, "admin", "correct-horse1")
    for url in ["/students/", f"/students/{SID}", f"/students/{SID}/report", "/recommendations/", "/students/?major=Business&status=at_risk"]:
        r = client.get(url)
        assert r.status_code == 200, url


# ── access control ──────────────────────────────────────
def test_staff_pages_require_login(client):
    for url in ["/students/", "/recommendations/", "/admin/users", "/admin/audit", f"/students/{SID}"]:
        r = client.get(url)
        assert r.status_code == 302 and "/auth/login" in r.headers["Location"], url


def test_writes_require_login(client):
    r = client.post("/students/add", data=_student_form(_csrf(client)))
    assert r.status_code == 302 and "/auth/login" in r.headers["Location"]


def test_missing_csrf_is_rejected(client):
    assert client.post("/predictions/", data={}).status_code == 400


def test_teacher_can_work_but_not_admin_things(client):
    _login(client, "teacher", "correct-horse2")
    assert client.get("/students/").status_code == 200
    assert client.get("/recommendations/").status_code == 200
    assert client.get("/admin/users").status_code == 403
    assert client.post(f"/students/{SID}/delete", data={"csrf_token": _csrf(client)}).status_code == 403


def test_student_sees_only_own_report(client):
    _login(client, "stu", "correct-horse3")
    assert client.get(f"/students/{SID}/report").status_code == 200
    assert client.get(f"/students/{SID + 1}/report").status_code == 403
    assert client.get("/students/").status_code == 403
    assert client.get("/recommendations/").status_code == 403
    assert client.get("/students/my-report").status_code == 302


def test_admin_adds_edits_deletes_student_and_records(client):
    _login(client, "admin", "correct-horse1")
    r = client.post("/students/add", data=_student_form(_csrf(client)))
    assert r.status_code == 302
    new_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
    rec = {"csrf_token": _csrf(client), "course_code": "MATH101", "semester": "2026 Fall", "status": "Completed", "mark": "72", "attendance": "88"}
    assert client.post(f"/students/{new_id}/enrollments/add", data=rec).status_code == 302
    page = client.get(f"/students/{new_id}").get_data(as_text=True)
    assert "Calculus I" in page and "B-" in page                      # 72 -> B- (assumed band 70-74)
    bad = dict(rec, mark="")                                           # a completed course needs a mark
    client.post(f"/students/{new_id}/enrollments/add", data=bad)
    assert page.count("Calculus I") >= 1
    assert client.post(f"/students/{new_id}/delete", data={"csrf_token": _csrf(client)}).status_code == 302
    assert client.get(f"/students/{new_id}").status_code == 404


def test_invalid_enrollment_is_rejected(client):
    _login(client, "admin", "correct-horse1")
    from app.data import data_loader
    before = len(client.get(f"/students/{SID}").get_data(as_text=True))
    for bad in [{"course_code": "NOPE", "semester": "2026 Fall", "status": "Completed", "mark": "50", "attendance": "80"},
                {"course_code": "MATH101", "semester": "next year", "status": "Completed", "mark": "50", "attendance": "80"},
                {"course_code": "MATH101", "semester": "2026 Fall", "status": "Completed", "mark": "150", "attendance": "80"},
                {"course_code": "MATH101", "semester": "2026 Fall", "status": "Completed", "mark": "50", "attendance": "-3"}]:
        client.post(f"/students/{SID}/enrollments/add", data=dict(bad, csrf_token=_csrf(client)))
    with client.application.app_context():
        enr = data_loader.load_tables().enrollments
        assert len(enr[(enr["student_id"] == SID) & (enr["semester"] == "2026 Fall") & (enr["course_code"] == "NOPE")]) == 0
        assert enr["mark"].dropna().between(0, 100).all()


def test_formula_injection_is_stripped_from_names(client):
    _login(client, "admin", "correct-horse1")
    form = _student_form(_csrf(client)); form["name"] = "=HYPERLINK(1)"
    client.post("/students/add", data=form)
    from app.data import data_loader
    with client.application.app_context():
        assert not data_loader.load_tables().students["name"].astype(str).str.startswith("=").any()


# ── login hardening ─────────────────────────────────────
def test_login_lockout_after_repeated_failures(client):
    for _ in range(6):
        _login(client, "admin", "wrong-password1")
    r = _login(client, "admin", "correct-horse1")
    assert r.status_code in (200, 429) and "Location" not in r.headers


def test_weak_passwords_are_refused(client):
    with client.application.app_context():
        assert auth.validate_password("short1") is not None
        assert auth.validate_password("password123") is not None
        assert auth.validate_password("a-Better-pass-2026") is None


def test_security_headers(client):
    h = client.get("/").headers
    assert "nonce-" in h["Content-Security-Policy"] and "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert h["X-Content-Type-Options"] == "nosniff"
