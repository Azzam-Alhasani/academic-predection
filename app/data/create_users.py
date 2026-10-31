"""
Seed user accounts. Run once:

    python app/data/create_users.py

It creates `app/data/users.csv` with a default admin account.
Passwords are hashed with Werkzeug and only the hash is stored.

By default it prints a random admin password ONCE so you can log in
and then change it. You can also pass a password as an argument.
"""

import os
import sys
import secrets

# Allow running from the project root or the app/data directory directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app import auth
from app.auth import add_user


def main():
    username = "admin"
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        # random, printed once below; retry until it satisfies the password policy
        from flask import Flask
        policy_app = Flask(__name__)
        while True:
            password = secrets.token_urlsafe(12)
            with policy_app.app_context():
                if auth.validate_password(password, production=True) is None:
                    break

    with __import__("flask").Flask(__name__).app_context():
        problem = auth.validate_password(password, production=False)
    if problem:
        sys.exit(f"Password refused: {problem}")
    add_user(username, password, role="admin")
    print(f"User '{username}' created.")
    print(f"Password: {password}")
    print("Store this password securely and change it after first login.")


if __name__ == "__main__":
    main()