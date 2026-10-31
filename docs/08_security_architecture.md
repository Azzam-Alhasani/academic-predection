# Security architecture

**Reviewer comment:** "Local server is used, reducing security."

## Response
A laptop server has no HTTPS, no access control and is reachable only on one machine. The system now runs as a hosted web service with layered controls.

| Layer | Control | Where |
|---|---|---|
| Transport | HTTPS terminated by the host, HSTS, proxy headers trusted only when `BEHIND_PROXY` is set | `app/__init__.py`, `config.py` |
| Identity | hashed passwords, strength policy, lockout by IP and username, constant-time checks | `app/auth.py` |
| Authorisation | admin / teacher / student, checked per route; `can_view_student` limits students to their own record | `app/auth.py`, `app/routes/` |
| Session | `session.clear()` on login, 30-minute idle expiry, HttpOnly + SameSite + Secure cookies | `config.py`, `app/auth.py` |
| Request integrity | CSRF token on every POST | `app/security.py` |
| Browser | CSP with nonce, no inline handlers, frame blocking, `no-store` for personal pages | `app/__init__.py` |
| Input | whitelist ranges/choices, 64 KB body limit, formula-character stripping | `app/routes/students.py` |
| Accountability | JSON-lines audit log, viewable by admin | `app/audit.py` |
| Secrets | environment variables only | `render.yaml` |

## Public vs protected
Public: dashboard, analytics, risk calculator, model-selection page (aggregates or stateless). Login required: student list/report, recommendations, edits. Admin only: accounts, audit log, deletes.

## Limits (be honest in the write-up)
- Data lives in CSV files; on Render's free plan edits are lost on restart. A managed database is the next step.
- Single-process rate limiting (in memory); no multi-factor authentication.
- The sample data is anonymised public data. Real student records need approval from the university and a private deployment.
- No independent penetration test has been done.
