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

from app.auth import add_user


def main():
    username = "admin"
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = secrets.token_urlsafe(12)  # random, printed once below

    add_user(username, password, role="admin")
    print(f"User '{username}' created.")
    print(f"Password: {password}")
    print("Store this password securely and change it after first login.")


if __name__ == "__main__":
    main()