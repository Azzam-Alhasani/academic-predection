"""
Flask application factory for the Academic Performance Monitoring & Prediction System.
"""

import os
from flask import Flask, request
from config import get_config


def create_app(config_name=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    # Ensure required directories exist
    os.makedirs(app.config["MODEL_DIR"], exist_ok=True)

    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.students import students_bp
    from app.routes.analytics import analytics_bp
    from app.routes.predictions import predictions_bp
    from app.routes.recommendations import recommendations_bp
    from app.auth import auth_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(students_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(auth_bp)

    # ── CSRF protection for all state-changing requests ──────
    from app.security import validate_csrf

    @app.before_request
    def _csrf_protect():
        # Skip CSRF for safe methods (GET/HEAD/OPTIONS). POST is the only
        # state-changing method used by this project's forms.
        if request.method == "POST":
            validate_csrf()

    # ── Template helpers (available in every template) ───────
    from app.security import get_csrf_token
    from app.auth import current_user

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": get_csrf_token,
            "current_user": current_user(),
        }

    return app