# Amastan Fraud Shield Guard — Development Roadmap

## Priority 1: Strengthen the Verified Deployment Path (Current)

### 1.1 Expand Test Coverage [DONE]
- **59 API tests** covering all endpoints (health, auth, alerts, feedback, stats, compliance-kpis, explain, drift, high-risk, branches, model-performance, export, ctaf-export, monitoring, retrain, edge cases, integration flow, legacy aliases)
- **153 tests total** across the project, all passing

### 1.4 Fix Deprecation Warnings [DONE]
- `datetime.utcnow()` → `datetime.now(timezone.utc).replace(tzinfo=None)` — 28 occurrences fixed across 9 files

### 1.2 Add Missing API Endpoints [DONE]
- Gap analysis: zero missing endpoints — dashboard calls all match existing routes
- Added `POST /api/v1/feedback/batch/` with legacy alias for bulk feedback submission
- Added `_links` HATEOAS to whoami, stats, CTAF export, and feedback responses
- 8 new tests covering batch feedback and HATEOAS links, all passing

### 1.3 Documentation Improvements
- `docs/` directory: add API reference, operational runbook, deployment guide
- Clarify the HF Space vs local Docker split in a dedicated DEPLOYMENT.md
- Add docstrings to all API endpoints (Pydantic models already have field_validator)
- Auto-generate OpenAPI spec, check all endpoints render correctly in Swagger

### 1.5 Additional Deprecation Cleanup
- SQLAlchemy datetime adapter deprecation
- Clean up httpx raw bytes warning noise

---

## Priority 2: Improve ML Pipeline

### 2.1 Champion/Challenger Workflow
- Verify `FraudModelTrainer.train_champion_challenger()` actually works end-to-end
- Add integration test: insert feedback → trigger retrain → verify champion promoted
- Add shadow model comparison tracking (already exists in `src/ml/shadow_model.py` — test it)

### 2.2 ML Metrics & Monitoring
- Verify `model_f1_score`, `model_auc` Prometheus metrics actually get set somewhere
- Build a retraining dashboard panel in the existing Grafana dashboard
- Add model drift detection as a scheduled background task in the API

### 2.3 Backtesting
- `scripts/backtest.py` — run it, verify output, add CI validation
- Compare champion vs challenger on historical data automatically

---

## Priority 3: CI/CD Pipeline

### 3.1 GitHub Actions
- Only `security-scan.yml` exists — add a full CI workflow:
  - `pip install` → `pytest tests/` (excluding chaos + quality_gates)
  - `bandit` + `pip-audit` + `safety`
  - Coverage report upload
- Dependabot already configured — verify it works

### 3.2 Quality Gates in CI
- Enforce test pass before merge
- Enforce minimum coverage (start at 70%)
- Lint check (flake8 + black + isort)

---

## Priority 4: Infrastructure Hardening

### 4.1 DLQ Log Noise
- Silence "No dead letter queue database found" message when no DLQ exists
- Add optional flag to disable DLQ retry worker

### 4.2 Root Route
- [DONE] `/` now returns API status — verified by test

### 4.3 Kubernetes Manifests
- Keep `k8s/` as-is. It is target architecture / portfolio signal, not active deployment.
- Do NOT invest time making K8s work as a runtime unless a cluster is provisioned.

---

## Priority 5: Observability Enhancements

### 5.1 Grafana Cloud Dashboard
- [DONE] `hosted_api_persistence.json` imported and working
- Add dashboard panels for model metrics when ML pipeline is verified

### 5.2 Local Monitoring Stack
- Verify `docker-compose.monitoring.yml` works end-to-end with the main stack
- Test alert rule firing locally

---

## Rules for This Roadmap

1. **Never change scope mid-priority** — finish Priority 1 entirely before touching Priority 2.
2. **Each task must produce a test** — no code change without a test that proves it works.
3. **Deprecation fixes are high-signal** — they remove noise and make real issues visible.
4. **If something is broken, first write a test that fails, then fix.**
5. **K8s is read-only.** Do not build, deploy, or maintain it.

---

## Session Log

| Date | Work Done |
|------|-----------|
| 2026-05-28 | Roadmap created. Task 1.1: expanded API test coverage — 153 tests total, all passing. Task 1.4: fixed 28 `datetime.utcnow()` deprecation warnings across 9 files. Task 1.2: no missing endpoints found, added batch feedback endpoint, added HATEOAS `_links` to responses, 8 new tests (161 total). |
