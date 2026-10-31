"""
Append-only audit trail: who did what, when and from where.

One JSON object per line in app/data/audit.log (not committed to Git). Passwords and
other secrets are never written. Admins can read the latest entries at /admin/audit.
"""

import json
import os
import threading
import time

from flask import request, session

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audit.log")
MAX_LINES = 5000          # older entries are dropped when the file grows past this
_lock = threading.Lock()


def log_event(action: str, detail: str = "", user: str | None = None) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
        "user": user or session.get("username") or "-",
        "ip": request.remote_addr if request else "-",
        "action": action,
        "detail": str(detail)[:200],
    }
    try:
        with _lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            _trim()
    except OSError:
        pass          # auditing must never break the request


def _trim() -> None:
    try:
        if os.path.getsize(LOG_FILE) < MAX_LINES * 120:
            return
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()[-MAX_LINES // 2:]
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except OSError:
        pass


def recent_events(limit: int = 200) -> list[dict]:
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events
