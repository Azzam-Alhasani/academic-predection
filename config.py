import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    DEBUG = False
    TESTING = False
    DATA_FILE = os.path.join(BASE_DIR, "app", "data", "students.csv")
    SAMPLE_DATA_FILE = os.path.join(BASE_DIR, "app", "data", "sample_students.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "app", "ml", "saved_models")
    HOST = "127.0.0.1"
    PORT = 5000


class DevelopmentConfig(Config):
    DEBUG = True
    SECRET_KEY = os.environ.get("SECRET_KEY", "apms-dev-secret-key-2026")


class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")

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