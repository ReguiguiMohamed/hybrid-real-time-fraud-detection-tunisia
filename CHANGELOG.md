# Changelog

All notable changes to this prototype are documented here.

## [0.1.0] - 2026-06-10

### Added

- Explicit prototype release scope and shared runtime version.
- Generated OpenAPI snapshot and deployment decision guide.
- SQLAlchemy model lifecycle repository for SQLite/PostgreSQL portability.
- Observable retraining job status.
- Champion model registry metrics.
- Backtest, model repository, trainer policy, and shadow comparison tests.
- Comprehensive finalization TODO.

### Changed

- CI now enforces formatting, static correctness, verified-slice coverage, and high-severity Bandit findings.
- Weekly dependency and Semgrep scans are explicitly advisory reports.
- Backtesting uses centralized prototype thresholds and honors the configured alert threshold.
- Shadow agreement no longer claims that a model is safe to promote.
- PostgreSQL deployments disable local Spark retraining and the SQLite DLQ worker by default.
- Packaging metadata moved to `pyproject.toml`.

### Fixed

- Migration checksums now use SHA-256 instead of MD5.
- Developer dependencies include the optional integration-test packages used during collection.
- Runtime dependencies explicitly include Faker for transaction generation.
- Champion metrics are populated from the shared model registry.

### Scope

The verified hosted release is the FastAPI command-center API with SQLAlchemy
persistence. The broader streaming and RAG architecture remains a local or
target-architecture prototype.
