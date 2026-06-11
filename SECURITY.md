# Security

Supported version: `0.1.x`

## Report A Problem

Do not open a public issue for a security problem.

Email `security@amastan.tn` with:

- the affected endpoint;
- steps to reproduce;
- expected impact;
- a suggested fix, if you have one.

## Current Controls

- Admin and analyst bearer tokens.
- SHA-256 token comparison.
- Pydantic request validation.
- Configurable CORS origins.
- Per-IP rate limiting.
- SQLAlchemy parameterized queries.
- Audit records for feedback and model changes.
- A protected Prometheus endpoint.
- Bandit in required CI.
- Weekly pip-audit and Semgrep reports.

## Known Limits

This is a portfolio prototype.

- Tokens are static secrets.
- Rate limiting is in memory.
- Tables are created on startup.
- SQLite is for local use only.
- The hosted database still needs an approved backup, retention, and migration
  plan.

## Before A Public Demo

Rotate:

- `ADMIN_TOKEN`
- `ANALYST_TOKEN`
- `METRICS_TOKEN`
- `HF_TOKEN`
- the database password

Use random values. Store them only in the relevant secret manager.
