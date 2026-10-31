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
    return {"csrf_token": token, "name": "Test Student", "course": "Mathematics", "t1_mark": 62, "t2_mark": 58,
            "absences": 4, "final_mark": ""}


# ── grading ─────────────────────────────────────────────
def test_grade_scale():
    assert grading.letter_for(90) == "A+" and grading.letter_for(50) == "D" and grading.letter_for(49.9) == "F"
    assert grading.points_for(75) == 3.0


def test_gpa_is_the_average_of_term_points():
    assert grading.gpa(75, 50) == 2.0                    # (3.0 + 1.0) / 2
    assert grading.gpa(75, None) == 3.0                  # term 2 not yet done
    assert grading.gpa(None, None) is None
    assert grading.average_mark(60, 50) == 55.0


def test_required_final_mark_formula():
    needed, reachable = grading.required_final_mark(50, 60, 60)      # 3 x 60 - 50 - 60
    assert needed == 70 and reachable
    assert grading.required_final_mark(50, 60, 100)[1] is False       # would need more than 100


def test_absence_bands():
    assert grading.absence_band(0) == "No absences" and grading.absence_band(12) == "11+ absences"


# ── pages and data ──────────────────────────────────────
def test_public_pages_load(client):
    for url in ["/", "/analytics/", "/predictions/", "/predictions/selection", "/auth/login", "/healthz"]:
        assert client.get(url).status_code == 200, url


def test_dataset_is_the_real_uci_data(client):
    from app.data.data_loader import load_students
    with client.application.app_context():
        df = load_students()
    assert len(df) == 1044                      # 395 Mathematics + 649 Portuguese records
    assert df["person_id"].nunique() == 662     # 382 students appear in both courses
    assert set(df["result"]) == {"Pass", "Fail"}
    assert df["gpa"].between(0, 4).all()
    extras = {"age", "gender", "scholarship", "tuition_paid", "debtor", "programme"}
    assert not extras & set(df.columns)         # only marks, absences and course are kept


def test_calculator_returns_risk_and_calculations(client):
    token = _csrf(client)
    form = _student_form(token)
    form.pop("name"); form.pop("final_mark")
    html = client.post("/predictions/", data=form).get_data(as_text=True)
    assert "GPA" in html and "Risk of failing" in html and "alert-danger" not in html


def test_calculator_rejects_bad_input(client):
    form = _student_form(_csrf(client)); form["t1_mark"] = 150
    assert "between 0 and 100" in client.post("/predictions/", data=form).get_data(as_text=True)


def test_calculator_works_without_term_2(client):
    form = _student_form(_csrf(client)); form.pop("name"); form.pop("final_mark"); form["t2_mark"] = ""
    html = client.post("/predictions/", data=form).get_data(as_text=True)
    assert "after term 1" in html and "alert-danger" not in html


def test_model_study_is_present_and_honest():
    from app.ml.predictor import load_selection
    study = load_selection()
    for scen in ("after_term1", "after_term2"):
        block = study["classification"][scen]
        assert len(block["candidates"]) >= 10 and block["selected"]
        base = next(c for c in block["candidates"] if c["key"] == "majority")
        best = next(c for c in block["tuned"] if c["key"] == block["selected"])
        assert best["metrics"]["accuracy"]["mean"] >= 0.85                       # the target accuracy
        assert best["metrics"]["accuracy"]["mean"] > base["metrics"]["accuracy"]["mean"] + 0.05
        assert study["regression"][scen]["selected"]


def test_no_leakage_between_a_students_courses():
    from app.ml.model_selection import grouped_splits
    import numpy as np
    groups = np.repeat(np.arange(100), 2)                    # every student has two records
    y = np.tile([0, 1], 100)
    for train, test in grouped_splits(y, groups, 5, 2, 1):
        assert not set(groups[train]) & set(groups[test])


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
