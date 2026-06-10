# Amastan Fraud Shield Guard - Task Automation
# Usage: make <target>
# All targets are idempotent and safe to re-run.

.PHONY: help setup setup-dev setup-ci dev prod test test-unit lint format clean build monitor monitor-down bootstrap deploy k8s-apply k8s-dry-run migrate migrate-status cost-estimate audit-deps chaos-test backtest shadow-register shadow-status security-scan circuit-status openapi

# Default target
help:
	@echo "============================================================"
	@echo "  Amastan Fraud Shield Guard - Operational Commands"
	@echo "============================================================"
	@echo ""
	@echo "  SETUP & BOOTSTRAP"
	@echo "  make setup              Initialize environment, install deps"
	@echo "  make setup-dev          Install runtime + development dependencies"
	@echo "  make setup-ci           Install the verified-slice CI dependencies"
	@echo "  make bootstrap          Seed database + train initial model"
	@echo "  make bootstrap-imbalanced  Realistic 0.01% fraud rate data"
	@echo "  make migrate            Run database schema migrations"
	@echo "  make migrate-status     Show migration status"
	@echo ""
	@echo "  DEVELOPMENT"
	@echo "  make dev                Start local dev stack (no Docker)"
	@echo "  make test               Run full test suite"
	@echo "  make test-unit          Run unit tests only (no Spark)"
	@echo "  make lint               Run linter + type checks"
	@echo "  make format             Format maintained Python sources"
	@echo "  make chaos-test         Run chaos/failure integration tests"
	@echo ""
	@echo "  PRODUCTION"
	@echo "  make prod               Start full Docker Compose stack"
	@echo "  make prod-monitor       Start Docker + Prometheus/Grafana"
	@echo "  make monitor            Start Prometheus + Grafana only"
	@echo "  make monitor-down       Stop monitoring stack only"
	@echo "  make build              Build all Docker images"
	@echo "  make deploy             Push images and deploy (K8s)"
	@echo ""
	@echo "  KUBERNETES"
	@echo "  make k8s-apply          Apply K8s manifests to cluster"
	@echo "  make k8s-dry-run        Dry-run K8s manifests (client-side)"
	@echo "  make k8s-delete         Delete all Amastan resources"
	@echo ""
	@echo "  BACKTESTING & MODEL OPS"
	@echo "  make backtest           Run backtest against historical data"
	@echo "  make shadow-register    Register a shadow model for comparison"
	@echo "  make shadow-status      Check current shadow model status"
	@echo "  make circuit-status     Check RAG circuit breaker status"
	@echo "  make openapi            Regenerate docs/openapi.json"
	@echo ""
	@echo "  COMPLIANCE & AUDIT"
	@echo "  make security-scan      Run bandit + pip-audit + safety"
	@echo "  make audit-deps         Check dependency pins and CVEs"
	@echo "  make cost-estimate      Calculate cloud infrastructure cost"
	@echo "  make clean              Remove all generated artifacts"
	@echo ""

# ==========================================
# Setup & Bootstrap
# ==========================================

setup:
	@echo "==> Installing dependencies from pinned requirements..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e . --no-deps
	@echo "==> Creating required directories..."
	mkdir -p data/parquet data/reports data/vector_db data/knowledge_base tmp/checkpoint models/registry
	@echo "==> Setup complete."

setup-dev:
	@echo "==> Installing runtime and development dependencies..."
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e . --no-deps
	@echo "==> Development environment ready."

setup-ci:
	@echo "==> Installing verified-slice CI dependencies..."
	pip install --upgrade pip
	pip install -r requirements-ci.txt
	pip install -e . --no-deps
	@echo "==> CI environment ready."

bootstrap:
	@echo "==> Bootstrapping fraud detection system..."
	@echo "==> Generating initial model and seeding database..."
	python scripts/bootstrap_system.py
	@echo "==> Running initial migration..."
	make migrate
	@echo "==> Bootstrap complete. System ready for development."

bootstrap-imbalanced:
	@echo "==> Bootstrapping with REALISTIC imbalanced data (0.01% fraud rate)..."
	python scripts/bootstrap_imbalanced.py --n-samples 100000 --fraud-rate 0.0001
	@echo "==> Running initial migration..."
	make migrate
	@echo "==> Imbalanced bootstrap complete. Model metrics are now meaningful."

# ==========================================
# Development
# ==========================================

dev:
	@echo "==> Starting local development stack..."
	@echo "==> API: http://localhost:8001"
	@echo "==> Dashboard: http://localhost:8501"
	@echo "==> Docs: http://localhost:8001/docs"
	@echo ""
	@echo "==> Starting API server..."
	uvicorn dashboard.api:app --host 0.0.0.0 --port 8001 --reload &
	@echo "==> Starting producer..."
	PYTHONPATH=src python src/producer/producer.py --rate 2 &
	@echo "==> Starting Streamlit dashboard..."
	streamlit run dashboard/dashboard.py &
	@echo ""
	@echo "==> Dev stack running. Press Ctrl+C to stop all background jobs."
	wait

test:
	@echo "==> Running full test suite..."
	pytest tests/ -v --tb=short --cov=src --cov=dashboard --cov-report=term-missing --cov-report=html:htmlcov
	@echo "==> Coverage report: htmlcov/index.html"

test-unit:
	@echo "==> Running unit tests (no Spark/Java required)..."
	pytest tests/test_api.py tests/test_schemas.py tests/test_producer.py tests/test_risk_config.py tests/test_utils.py tests/test_monitoring.py -v --tb=short

lint:
	@echo "==> Running static correctness checks..."
	flake8 src/ dashboard/ scripts/ tests/
	@echo "==> Running black (check mode)..."
	black --check src/ dashboard/ scripts/ tests/
	@echo "==> Running isort (check mode)..."
	isort --check-only src/ dashboard/ scripts/ tests/
	@echo "==> Linting complete."

format:
	@echo "==> Formatting maintained Python sources..."
	isort src/ dashboard/ scripts/ tests/
	black src/ dashboard/ scripts/ tests/
	@echo "==> Formatting complete."

chaos-test:
	@echo "==> Running chaos/integration tests..."
	pytest tests/test_chaos_integration.py -v --tb=short
	@echo "==> Chaos tests complete."

# ==========================================
# Production
# ==========================================

prod:
	@echo "==> Starting full production stack..."
	docker compose up --build -d
	@echo "==> Waiting for services to be healthy..."
	@echo "==> API: http://localhost:8001"
	@echo "==> Dashboard: http://localhost:8501"
	@echo "==> ChromaDB: http://localhost:8000"
	@echo ""
	docker compose ps

prod-monitor:
	@echo "==> Starting full stack with monitoring..."
	docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up --build -d
	@echo "==> Waiting for services to be healthy..."
	@echo "==> API: http://localhost:8001"
	@echo "==> Dashboard: http://localhost:8501"
	@echo "==> Prometheus: http://localhost:9090"
	@echo "==> Grafana: http://localhost:3000 (admin/admin)"
	@echo "==> Alertmanager: http://localhost:9093"
	@echo ""
	docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml ps

monitor:
	@echo "==> Starting monitoring stack only..."
	docker compose -f monitoring/docker-compose.monitoring.yml up -d
	@echo "==> Prometheus: http://localhost:9090"
	@echo "==> Grafana: http://localhost:3000 (admin/admin)"
	@echo "==> Alertmanager: http://localhost:9093"

monitor-down:
	@echo "==> Stopping monitoring stack..."
	docker compose -f monitoring/docker-compose.monitoring.yml down

build:
	@echo "==> Building all Docker images..."
	docker compose build --parallel

deploy:
	@echo "==> Deploying to Kubernetes cluster..."
	@echo "==> Pushing images to registry..."
	docker compose push
	@echo "==> Applying K8s manifests..."
	kubectl apply -f k8s/namespace.yml
	kubectl apply -f k8s/
	@echo "==> Deployment in progress. Monitor with: kubectl get pods -n amastan"

# ==========================================
# Kubernetes
# ==========================================

k8s-apply:
	@echo "==> Applying Kubernetes manifests..."
	kubectl apply -f k8s/namespace.yml
	kubectl apply -f k8s/
	@echo "==> Manifests applied."

k8s-dry-run:
	@echo "==> Dry-run: validating Kubernetes manifests locally..."
	python scripts/validate_k8s_manifests.py
	@echo "==> Validation complete."

k8s-delete:
	@echo "==> WARNING: Deleting all Amastan K8s resources..."
	kubectl delete -f k8s/
	kubectl delete -f k8s/namespace.yml
	@echo "==> Resources deleted."

# ==========================================
# Database Migrations
# ==========================================

migrate:
	@echo "==> Running database migrations..."
	python scripts/migrate.py upgrade
	@echo "==> Migrations applied."

migrate-status:
	@echo "==> Checking migration status..."
	python scripts/migrate.py current
	@echo "==> Status displayed."

migrate-generate:
	@echo "==> Generating new migration from current state..."
	@read -p "Enter migration description: " desc; \
	python scripts/migrate.py migrate "$$desc"
	@echo "==> Migration generated."

# ==========================================
# Compliance & Audit
# ==========================================

audit-deps:
	@echo "==> Auditing pinned dependencies..."
	python scripts/audit_dependencies.py

cost-estimate:
	@echo "==> Cloud Infrastructure Cost Estimation..."
	python scripts/cost_estimate.py

# ==========================================
# Cleanup
# ==========================================

clean:
	@echo "==> WARNING: This will remove all generated data, models, and checkpoints."
	@echo "==> Cleaning..."
	rm -rf data/parquet data/reports tmp/checkpoint tmp/spark-warehouse tmp/checkpoint_stateful
	rm -rf data/dedup_cache.db
	rm -rf models/registry
	rm -rf htmlcov .pytest_cache .coverage pytest-cache-files-*
	rm -rf data/pytest-cache-files-* data/pytest_basetemp* data/pytest_tmp
	rm -f backtest_report.json notebooks/output_*.html
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Clean complete. Source code and .env files preserved."

# ==========================================
# Backtesting & Model Ops
# ==========================================

backtest:
	@echo "==> Running backtest against historical data..."
	python scripts/backtest.py --output backtest_report.json
	@echo "==> Report saved to backtest_report.json"

shadow-register:
	@echo "==> Register shadow model for comparison..."
	@read -p "Enter model path: " path; \
	python -c "from src.ml.shadow_model import ShadowModelManager; s=ShadowModelManager(); s.register_shadow_model('$$path'); print(f'Shadow model registered from $$path')"
	@echo "==> Shadow model is now scoring alongside champion (alerts NOT triggered)"

shadow-status:
	@echo "==> Shadow model status..."
	python -c "from src.ml.shadow_model import ShadowModelManager; import json; s=ShadowModelManager(); print(json.dumps(s.get_shadow_status(), indent=2, default=str))"

circuit-status:
	@echo "==> RAG Circuit Breaker Status..."
	python -c "from src.rag_engine.circuit_breaker import get_rag_circuit; import json; c=get_rag_circuit(); print(json.dumps(c.get_stats(), indent=2))"

openapi:
	@echo "==> Generating OpenAPI spec..."
	python scripts/generate_openapi.py
	@echo "==> OpenAPI spec generated: docs/openapi.json"

# ==========================================
# Security Scanning
# ==========================================

security-scan:
	@echo "==> Running Bandit (Python security linter)..."
	@bandit -r src/ dashboard/ scripts/ -f txt || true
	@echo ""
	@echo "==> Running pip-audit (CVE scanner)..."
	@pip-audit --requirement requirements.txt || true
	@echo ""
	@echo "==> Running Safety (vulnerability check)..."
	@safety check -r requirements.txt || true
	@echo ""
	@echo "==> Security scan complete. Review output above for findings."
