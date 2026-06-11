# Changelog

All notable changes to this prototype are documented here.

## [0.1.0] - 2026-06-11

### Added

- Explicit prototype release scope and shared runtime version.
- Generated OpenAPI snapshot and deployment decision guide.
- SQLAlchemy model lifecycle repository for SQLite/PostgreSQL portability.
- Observable retraining job status.
- Champion model registry metrics.
- Backtest, model repository, trainer policy, and shadow comparison tests.
- Comprehensive finalization TODO.

### Changed

- Marked the portfolio prototype complete and moved the repository to maintenance mode.
- Removed links to internal planning documents from public documentation.
- Updated pinned GitHub Actions to Node.js 24-compatible releases.
- CI now enforces formatting, static correctness, verified-slice coverage, and high-severity Bandit findings.
- CI now fails when deterministic backtest artifact generation fails.
- Weekly dependency and Semgrep scans are explicitly advisory reports.
- Backtesting uses centralized prototype thresholds and honors the configured alert threshold.
- Shadow agreement no longer claims that a model is safe to promote.
- PostgreSQL deployments disable local Spark retraining and the SQLite DLQ worker by default.
- Packaging metadata moved to `pyproject.toml`.

### Fixed

- Backtest artifact generation now works from any caller working directory.
- Corrected supported-version, security-scanner, and token-rotation documentation.
- Migration checksums now use SHA-256 instead of MD5.
- Developer dependencies include the optional integration-test packages used during collection.
- Runtime dependencies explicitly include Faker for transaction generation.
- CI installs only dependencies required by the enforced prototype slice.
- Optional ML, RAG, tracing, and infrastructure packages load on demand.
- Champion metrics are populated from the shared model registry.

### Scope

The verified hosted release is the FastAPI command-center API with SQLAlchemy
persistence. The broader streaming and RAG architecture remains a local or
target-architecture prototype.

## [0.1.0-prototype] - 2026-06-10

### Added

- Training failure events recorded in lifecycle audit trail (`TRAINING_FAILURE` action).
- `last_training_success_at`, `last_training_failure_at`, `last_training_error` columns on `ModelRegistry`.
- `GET /api/v1/retrain-model/summary` endpoint exposing persistent training status.
- `ModelRepository.record_training_outcome()` and `get_champion_training_status()` methods.
- Tests: champion artifact load failure, training failure audit trail, backtest empty input/fallback.
- Accepted dependency risks table with expiry dates in SECURITY.md.

### Changed

- `dashboard/api.py`: ANALYST_TOKEN and ADMIN_TOKEN now raise `RuntimeError` instead of defaulting.
- `train_champion_challenger()` wraps training in try/except; failures are logged as audit events and stored in registry.
- GitHub Actions pinned to specific commit SHAs (`actions/checkout@692973e3`, `actions/setup-python@39cd1854`, `actions/upload-artifact@65c4c4a1`).
- Regenerated `openapi.json` to match FastAPI 0.115.0 `ValidationError` schema (removed `input`/`ctx` fields).

### Removed

- Default development tokens from production API code.
- `pytest-cache-files-*` and `tmp/` directories from disk.
