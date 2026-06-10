# Prototype Finalization TODO

Target release: `v0.1.0-prototype`

The release boundary is deliberate: the FastAPI command-center API with
SQLAlchemy persistence is the verified hosted prototype. Kafka, Spark,
Ollama, ChromaDB, Streamlit, Prometheus, and Kubernetes remain local or
target-architecture components until separately deployed and verified.

## 1. Release Baseline

- [x] Recover repository context from Markdown, Git history, and handoff notes.
- [x] Verify the pending OpenAPI and deployment documentation work.
- [x] Run the API test suite.
- [x] Run the maintained test suite without optional container tests.
- [x] Commit the completed Priority 1 OpenAPI/deployment work.
- [x] Set one source of truth for the prototype version.
- [x] Add a release checklist and changelog.
- [ ] Create and push `v0.1.0-prototype`.
- [ ] Publish a GitHub prerelease with an explicit prototype scope.

## 2. Repository Organization

- [x] Replace duplicate packaging metadata with `pyproject.toml`.
- [x] Add a pinned development requirements file.
- [x] Keep generated runtime data and model artifacts out of Git.
- [ ] Remove obsolete imports and dead compatibility code where tests prove it unused.
- [x] Format maintained Python sources consistently.
- [x] Keep Kubernetes manifests read-only until a cluster is provisioned.
- [x] Preserve evidence images and exported Grafana dashboards.

## 3. Truthful CI

- [x] Make formatting checks fail when formatting is wrong.
- [x] Make static correctness checks fail on syntax and undefined-name errors.
- [x] Make the configured coverage threshold actually fail below 70%.
- [x] Install all test dependencies used by collected tests.
- [x] Separate required CI gates from advisory security reports.
- [ ] Pin GitHub Actions and development tool versions.
- [x] Add OpenAPI snapshot validation to CI.
- [x] Add focused backtest tests to the required CI suite.

## 4. API and Deployment Slice

- [x] Document Hugging Face + Neon versus local Docker deployment.
- [x] Generate and track the OpenAPI specification.
- [x] Test expected API paths, Swagger rendering, and legacy deprecation.
- [x] Use the prototype version in API discovery and health responses.
- [x] Disable the DLQ retry worker through explicit configuration.
- [x] Remove noisy DLQ startup output when no queue exists.
- [x] Return an observable retraining job status instead of hiding background failures.
- [x] Keep hosted metrics authentication fail-closed on PostgreSQL.

## 5. Persistence and Model Lifecycle

- [x] Replace SQLite-only retraining persistence with the shared SQLAlchemy database.
- [x] Support both local SQLite and hosted PostgreSQL for feedback and model registry access.
- [x] Keep model artifact storage distinct from model metadata storage.
- [x] Add a clear challenger state without silently replacing the champion.
- [x] Require an identified human approver for promotion.
- [ ] Record promotion, rejection, and training-failure events in one lifecycle audit trail.
- [ ] Test feedback insertion -> retraining -> challenger registration -> promotion.
- [x] Test behavior when feedback is insufficient.
- [ ] Test behavior when the current champion artifact cannot be loaded.

## 6. ML Metrics and Monitoring

- [x] Populate champion F1, AUC, and version metrics from the model registry.
- [ ] Expose retraining result and last-success timestamps.
- [ ] Add model lifecycle panels to the local Grafana dashboard.
- [ ] Keep hosted API metrics separate from undeployed pipeline metrics.
- [x] Add tests for model metric refresh behavior.
- [x] Make drift scheduling explicit and configurable.

## 7. Shadow Model Workflow

- [ ] Use the shared database abstraction instead of direct SQLite connections.
- [x] Test shadow registration and replacement.
- [x] Test comparison recording.
- [ ] Join shadow predictions to analyst labels before claiming performance.
- [x] Distinguish agreement from superiority in recommendations.
- [ ] Record latency and sample counts with comparison metrics.

## 8. Rules and Backtesting

- [x] Centralize prototype risk thresholds and weights.
- [x] Remove stale hard-coded TND 5,000 and TND 2,000 values from backtesting.
- [x] Make `alert_threshold` affect both original and modified backtest results.
- [x] Require real labels for deployment recommendations.
- [x] Label heuristic/proxy evaluations as non-decisional.
- [ ] Extend backtest coverage to model-artifact fallback and empty input data.
- [ ] Produce a deterministic backtest JSON artifact in CI.

## 9. Documentation and Claims

- [x] State prototype maturity near the top of the README.
- [x] Remove claims that exceed the verified deployment evidence.
- [x] Describe configurable monitoring rules as prototype controls, not regulation.
- [x] Align the model card with the actual training and deployment path.
- [x] Add a concise design philosophy without turning documentation into branding copy.
- [x] Link the API reference, deployment guide, runbook, roadmap, and TODO.
- [x] Update the GitHub description and repository topics.

## 10. Security and Operations

- [x] Fail CI on high-confidence, high-severity Bandit findings.
- [x] Publish dependency audit artifacts even when dependency remediation is deferred.
- [ ] Document accepted dependency risks with expiry dates.
- [ ] Remove development default tokens from production deployments.
- [ ] Verify CORS and rate-limit configuration for the hosted API.
- [ ] Verify database migration behavior against PostgreSQL.
- [ ] Rotate deployment and metrics tokens before a public demonstration.

## 11. Final Verification

- [x] `python -m compileall src dashboard scripts`
- [x] `black --check src dashboard scripts tests`
- [x] `isort --check-only src dashboard scripts tests`
- [x] Static correctness check passes.
- [x] Unit/API tests pass with at least 70% configured coverage.
- [x] Optional chaos tests either pass with Docker or skip with a documented reason.
- [x] OpenAPI snapshot regenerates without a diff.
- [x] Local Docker build explicitly deferred; Hugging Face is the release target.
- [x] Hugging Face deployment builds and its health/version smoke test passes.
- [ ] Git worktree is clean after release commit.
- [ ] GitHub Actions pass on the tagged commit.
