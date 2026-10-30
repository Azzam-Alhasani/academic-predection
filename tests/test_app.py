"""Smoke tests: pages load, data is the real UCI set, writes need a login."""

import shutil

import pytest

import app.auth as auth
from app import create_app
from app.ml.preprocessing import FEATURES


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = create_app("development")
    app.config["TESTING"] = True
    data = tmp_path / "students.csv"
    shutil.copy(app.config["SAMPLE_DATA_FILE"], data)
    app.config["DATA_FILE"] = str(data)
    monkeypatch.setattr(auth, "_users_file", lambda: str(tmp_path / "users.csv"))
    auth._failed_logins.clear()
    auth.add_user("admin", "correct-horse", role="admin")
    return app.test_client()


def _csrf(client):
    with client.session_transaction() as s:
        s["_csrf_token"] = "test-token"
    return "test-token"


def _student_form(token):
    return {
        "csrf_token": token, "name": "Test Student", "subject": "Math", "gender": "Female",
        "higher": "yes", "internet": "yes", "age": 17, "studytime": 2, "failures": 0,
        "absences": 3, "G1": 11, "G2": 12, "G3": 12,
    }


def test_public_pages_load(client):
    for url in ["/", "/students/", "/analytics/", "/predictions/", "/auth/login", "/healthz"]:
        assert client.get(url).status_code == 200, url


def test_dataset_is_the_real_uci_data(client):
    from app.data.data_loader import load_students
    with client.application.app_context():
        df = load_students()
    assert len(df) == 1044                      # 395 Math + 649 Portuguese students
    assert set(df["subject"]) == {"Math", "Portuguese"}
    assert df["G3"].between(0, 20).all()


def test_final_grade_is_not_a_model_input(client):
    assert "G3" not in FEATURES and "pass_fail" not in FEATURES


def test_writes_require_login(client):
    token = _csrf(client)
    r = client.post("/students/add", data=_student_form(token))
    assert r.status_code == 302 and "/auth/login" in r.headers["Location"]
    assert client.get("/recommendations/").status_code == 302


def test_post_without_csrf_is_rejected(client):
    assert client.post("/students/add", data=_student_form("x")).status_code == 400


def test_login_then_add_student(client):
    token = _csrf(client)
    bad = client.post("/auth/login", data={"csrf_token": token, "username": "admin", "password": "nope"})
    assert b"Invalid username or password" in bad.data
    ok = client.post("/auth/login", data={"csrf_token": token, "username": "admin", "password": "correct-horse"})
    assert ok.status_code == 302
    r = client.post("/students/add", data=_student_form(token))
    assert r.status_code == 302 and "/students/1045" in r.headers["Location"]


def test_login_is_throttled(client):
    token = _csrf(client)
    codes = [client.post("/auth/login", data={"csrf_token": token, "username": "admin", "password": "x"}).status_code
             for _ in range(7)]
    assert codes[-1] == 429


def test_open_redirect_is_blocked(client):
    token = _csrf(client)
    r = client.post("/auth/login?next=//evil.example", data={"csrf_token": token, "username": "admin", "password": "correct-horse"})
    assert "evil.example" not in r.headers["Location"]
