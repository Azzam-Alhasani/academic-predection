"""
Flask application factory for the Academic Performance Monitoring & Prediction System.
"""

import os
import secrets

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from werkzeug.exceptions import HTTPException

from config import get_config


def create_app(config_name=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    # Accounts: ADMIN_PASSWORD creates the admin on hosts without a shell; in development a
    # default local account is created when none exists.
    from app.auth import ensure_admin_from_env, ensure_dev_admin
    ensure_admin_from_env(app)
    ensure_dev_admin(app)

    from app.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.analytics import analytics_bp
    from app.routes.main import main_bp
    from app.routes.predictions import predictions_bp
    from app.routes.recommendations import recommendations_bp
    from app.routes.students import students_bp

    for blueprint in (main_bp, students_bp, analytics_bp, predictions_bp, recommendations_bp, admin_bp, auth_bp):
        app.register_blueprint(blueprint)

    # ── CSRF protection for all state-changing requests ──────
    from app.security import validate_csrf

    @app.before_request
    def _csrf_protect():
        g.csp_nonce = secrets.token_urlsafe(16)
        if request.method == "POST":
            try:
                validate_csrf()
            except HTTPException:
                if request.endpoint == "auth.login":
                    # Usually an expired page or blocked cookies: explain instead of a bare 400.
                    flash("Your session expired or cookies are blocked. Please try signing in again.", "warning")
                    return redirect(url_for("auth.login"))
                raise

    app.jinja_env.filters["zip"] = lambda a, *others: zip(a, *others)

    # ── Template helpers (available in every template) ───────
    from app.auth import current_user, is_staff
    from app.security import get_csrf_token

    @app.context_processor
    def inject_globals():
        user = current_user()
        return {
            "csrf_token": get_csrf_token,
            "current_user": user,
            "is_staff": is_staff(user),
            "is_admin": bool(user) and user["role"] == "admin",
            "csp_nonce": g.get("csp_nonce", ""),
        }

    # ── Public hosting: trust the host's reverse proxy for client IP / HTTPS ──
    if app.config.get("BEHIND_PROXY"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    # ── Security headers ─────────────────────────────────────
    @app.after_request
    def _security_headers(response):
        nonce = g.get("csp_nonce", "")
        response.headers.setdefault("Content-Security-Policy", (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"))
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if session.get("username"):
            response.headers["Cache-Control"] = "no-store"      # personal data is not cached by browsers/proxies
        return response

    # ── Friendly error pages (no stack traces for visitors) ──
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(400)
    @app.errorhandler(429)
    def _client_error(error):
        messages = {
            400: ("Bad request", "The request could not be processed. Go back and try again."),
            403: ("Not allowed", "Your account does not have access to this page."),
            404: ("Page not found", "That page does not exist."),
            429: ("Too many requests", "Please wait a moment and try again."),
        }
        title, text = messages.get(error.code, ("Error", "Something went wrong."))
        return render_template("error.html", code=error.code, title=title, text=text), error.code

    @app.errorhandler(500)
    def _server_error(error):
        return render_template("error.html", code=500, title="Something went wrong",
                               text="The server hit an unexpected problem. It has not changed your data."), 500

    return app
