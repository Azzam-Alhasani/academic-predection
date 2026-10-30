import os
import secrets

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    DEBUG = False
    TESTING = False
    DATA_FILE = os.path.join(BASE_DIR, "app", "data", "students.csv")
    SAMPLE_DATA_FILE = os.path.join(BASE_DIR, "app", "data", "sample_students.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "app", "ml", "saved_models")
    HOST = "127.0.0.1"
    PORT = 5000

    # Cookie hardening (safe for localhost too).
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = False

    # Login throttling (per client IP): max failed attempts inside the window.
    LOGIN_MAX_ATTEMPTS = 5
    LOGIN_WINDOW_SECONDS = 300


class DevelopmentConfig(Config):
    DEBUG = True
    # A random key per start is fine for local development (sessions reset on restart),
    # unless SECRET_KEY is set in the environment / .env file.
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")
    HOST = "0.0.0.0"  # a hosting platform's proxy needs to reach the app
    SESSION_COOKIE_SECURE = True  # cookies only over HTTPS
    BEHIND_PROXY = True  # Render/Railway/Fly put a reverse proxy (HTTPS) in front of the app

    def __init__(self):
        if not self.SECRET_KEY:
            raise RuntimeError("SECRET_KEY must be set in production.")


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config(config_name=None):
    name = config_name or os.environ.get("FLASK_ENV", "default")
    return config.get(name, DevelopmentConfig)()
