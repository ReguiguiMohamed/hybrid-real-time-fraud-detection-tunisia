# Amastan Fraud Shield 0.1.0 Prototype

This prerelease establishes a truthful prototype boundary.

## Verified Slice

- FastAPI command-center API
- SQLite and PostgreSQL persistence through SQLAlchemy
- Authenticated analyst, administrator, and metrics surfaces
- Generated OpenAPI contract with 35 paths
- Observable local retraining jobs when explicitly enabled
- Champion model metrics sourced from the model registry

## Engineering Changes

- Replaced duplicate packaging metadata with `pyproject.toml`.
- Enforced formatting, static checks, coverage, OpenAPI drift, and severe
  Bandit findings in CI.
- Centralized prototype risk thresholds and removed duplicated backtest logic.
- Added a shared model lifecycle repository and explicit human promotion gate.
- Expanded focused tests for backtesting, model lifecycle, shadow comparison,
  and API behavior.

## Scope

Kafka, Spark, Streamlit, Ollama, ChromaDB, Prometheus/Grafana, and Kubernetes
remain local or target-architecture components. They are not represented as a
verified hosted production system.

The design rule is simple: a claim earns its place by surviving contact with a
test, a failure, or a deployed endpoint.
