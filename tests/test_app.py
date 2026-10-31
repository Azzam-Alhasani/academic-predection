"""Tests: pages load, UTAS grading maths, roles and access control, security headers, model study."""

import shutil

import pytest

import app.auth as auth
from app import create_app, grading


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = create_app("development")
    app.config["TESTING"] = True
    data = tmp_path / "students.csv"
    shutil.copy(app.config["SAMPLE_DATA_FILE"], data)
    app.config["DATA_FILE"] = str(data)
    monkeypatch.setattr(auth, "_users_file", lambda: str(tmp_path / "users.csv"))
    monkeypatch.setattr("app.audit._log_path", lambda: str(tmp_path / "audit.log"), raising=False)
    auth._failed_logins.clear()
    auth.add_user("admin", "correct-horse1", role="admin")
    auth.add_user("teacher", "correct-horse2", role="teacher")
    auth.add_user("stu", "correct-horse3", role="student", student_id=3)
    return app.test_client()


def _csrf(client):
    with client.session_transaction() as s:
        s["_csrf_token"] = "test-token"
    return "test-token"


def _login(client, username, password):
    return client.post("/auth/login", data={"username": username, "password": password, "csrf_token": _csrf(client)})


def _student_form(token):
    return {
        "csrf_token": token, "name": "Test Student", "gender": "Female", "programme": "Nursing",
        "attendance": "Daytime", "scholarship": "no", "tuition_paid": "yes", "debtor": "no", "age": 19,
        "admission_mark": 60, "prev_mark": 60, "s1_enrolled": 6, "s1_approved": 5, "s1_mark": 62,
        "s2_enrolled": 6, "s2_approved": 4, "s2_mark": 58, "outcome": "Enrolled",
    }


# ── grading ─────────────────────────────────────────────
def test_grade_scale():
    assert grading.letter_for(90) == "A+" and grading.letter_for(50) == "D" and grading.letter_for(49.9) == "F"
    assert grading.points_for(75) == 3.0


def test_cgpa_is_weighted_by_courses():
    assert grading.semester_gpa(75, 6, 6) == 3.0                     # mark 75 = B = 3.0, all passed
    assert grading.semester_gpa(75, 3, 6) == 1.5                     # only half the courses passed
    assert grading.cgpa(75, 6, 6, 50, 6, 6) == 2.0                   # (3.0 x 6 + 1.0 x 6) / 12
    assert grading.cgpa(0, 0, 0) is None


def test_required_gpa_formula():
    needed, reachable = grading.required_semester_gpa(2.0, 12, 2.5, 6)
    assert needed == pytest.approx((2.5 * 18 - 2.0 * 12) / 6) and reachable
    assert grading.required_semester_gpa(2.0, 12, 4.0, 1)[1] is False   # needs more than 4.0


# ── pages and data ──────────────────────────────────────
def test_public_pages_load(client):
    for url in ["/", "/analytics/", "/predictions/", "/predictions/selection", "/auth/login", "/healthz"]:
        assert client.get(url).status_code == 200, url


def test_dataset_is_the_real_uci_data(client):
    from app.data.data_loader import load_students
    with client.application.app_context():
        df = load_students()
    assert len(df) == 4424
    assert set(df["outcome"]) == {"Dropout", "Graduate", "Enrolled"}
    assert df["cgpa"].dropna().between(0, 4).all()


def test_calculator_returns_risk_and_calculations(client):
    token = _csrf(client)
    form = _student_form(token)
    form.pop("name"); form.pop("outcome")
    html = client.post("/predictions/", data=form).get_data(as_text=True)
    assert "CGPA" in html and "alert-danger" not in html


def test_calculator_rejects_bad_input(client):
    form = _student_form(_csrf(client)); form["s1_mark"] = 150
    assert "between 0 and 100" in client.post("/predictions/", data=form).get_data(as_text=True)


def test_model_study_is_present_and_honest():
    from app.ml.predictor import load_selection
    study = load_selection()
    for scen in ("after_sem1", "after_sem2"):
        block = study["classification"][scen]
        assert len(block["candidates"]) >= 10 and block["selected"]
        base = next(c for c in block["candidates"] if c["key"] == "majority")
        best = next(c for c in block["tuned"] if c["key"] == block["selected"])
        assert best["metrics"]["accuracy"]["mean"] > base["metrics"]["accuracy"]["mean"] + 0.1


# ── access control ──────────────────────────────────────
def test_staff_pages_require_login(client):
    for url in ["/students/", "/recommendations/", "/admin/users", "/admin/audit", "/students/1"]:
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
    assert client.post("/students/1/delete", data={"csrf_token": _csrf(client)}).status_code == 403


def test_student_sees_only_own_report(client):
    _login(client, "stu", "correct-horse3")
    assert client.get("/students/3/report").status_code == 200
    assert client.get("/students/4/report").status_code == 403
    assert client.get("/students/").status_code == 403
    assert client.get("/recommendations/").status_code == 403
    assert client.get("/students/my-report").status_code == 302


def test_admin_adds_edits_deletes_student(client):
    _login(client, "admin", "correct-horse1")
    r = client.post("/students/add", data=_student_form(_csrf(client)))
    assert r.status_code == 302
    new_id = int(r.headers["Location"].rstrip("/").split("/")[-1])
    assert client.get(f"/students/{new_id}").status_code == 200
    assert client.post(f"/students/{new_id}/delete", data={"csrf_token": _csrf(client)}).status_code == 302
    assert client.get(f"/students/{new_id}").status_code == 404


def test_formula_injection_is_stripped_from_names(client):
    _login(client, "admin", "correct-horse1")
    form = _student_form(_csrf(client)); form["name"] = "=HYPERLINK(1)"
    client.post("/students/add", data=form)
    from app.data.data_loader import load_students
    with client.application.app_context():
        assert not load_students()["name"].astype(str).str.startswith("=").any()


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
