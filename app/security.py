import hmac
import os

from flask import abort, request, session


def get_csrf_token() -> str:
    token = session.get("_csrf_token")
    if token is None:
        token = os.urandom(16).hex()
        session["_csrf_token"] = token
    return token


def validate_csrf() -> bool:
    stored = session.get("_csrf_token")
    if not stored:
        abort(400, description="Missing CSRF token.")

    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not submitted:
        abort(400, description="CSRF token required.")

    if not hmac.compare_digest(str(submitted), str(stored)):   # constant-time comparison
        abort(400, description="CSRF token mismatch.")

    return True
