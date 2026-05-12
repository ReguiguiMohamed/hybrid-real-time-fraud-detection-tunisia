---
title: Amastan Fraud Shield Guard
sdk: docker
app_port: 7860
---
# Amastan Fraud Shield Guard - Hybrid Streaming & RAG Architecture

A production-oriented real-time fraud mitigation prototype for Tunisian digital payments. Uses **Kafka + Spark Structured Streaming** for stateful detection, **XGBoost** for ML scoring, and **RAG (Ollama/ChromaDB)** for analyst-reviewed SAR drafting with deterministic compliance fallback.

Built for the 2026 Tunisian digital-payments landscape: digital usage is growing, but cash still matters and adoption is uneven outside major urban areas. The 2026 Finance Law repealed the TND 5,000 cash-payment cap, so this project avoids treating that old cap as a structuring rule and uses velocity, account, rail, sanctions/PEP, and analyst-review signals instead.

---

## Live Deployment Verification

The command-center API has been deployed on a Hugging Face Docker Space and connected to a Neon PostgreSQL database through SQLAlchemy. The screenshot below shows the deployed `/health/` endpoint, an authenticated Swagger request returning a persisted high-risk alert, and the same alert visible in Neon.

![Hugging Face API, Swagger readback, and Neon PostgreSQL verification](resultscreenshot.png)

This validates the free-tier deployment path used for the prototype: Hugging Face Spaces hosts the FastAPI service, Neon provides managed PostgreSQL persistence, and the same API can still fall back to local SQLite for development. It is a pragmatic demonstration setup, not a claim that the free-tier stack is sufficient for regulated production traffic.

Verified path:

`FastAPI on Hugging Face Spaces -> authenticated /api/v1 alert endpoint -> SQLAlchemy -> Neon PostgreSQL -> API readback`

---

## Verified Deployment Slice

The hosted Space currently proves the command-center persistence slice:

```mermaid
flowchart LR
    Client[Swagger / API Client] --> HF[Hugging Face Space\nFastAPI on port 7860]
    HF --> Auth[Bearer-token RBAC\nadmin / analyst]
    Auth --> SQLA[SQLAlchemy Session Layer]
    SQLA --> Neon[(Neon PostgreSQL\nhigh_risk_alerts + feedback tables)]
    Neon --> Readback[GET /api/v1/alerts/high-risk/]
```

The full Kafka/Spark/Ollama/Chroma stack below is the broader local and target
system architecture. The Hugging Face Space intentionally runs only the FastAPI
command-center API; stream processing, local SAR generation, monitoring, and
dashboard components run in the local Docker/Kubernetes-oriented stack.

---

## Full System Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '20px'}}}%%
flowchart LR
    Prod[Local Simulator\nDev/Test Events] --> Kafka[Kafka\ntunisian_transactions]
    Kafka --> CG[Consumer\nSpark SS + Stateful]

    subgraph Pipeline["Fraud Detection Pipeline"]
        CG --> QG[Quality Gates\nValidate + D17 Rules]
        QG --> SA[Stateful Aggregation\n5-min windows per user]
        SA --> RS[Risk Engine\nDynamic Rules]
        RS --> ML[ML Inference\nXGBoost Champion]
        ML --> AP[Alert Dispatch\nAsync + SAR]
    end

    AP --> API[FastAPI\nCommand Center]
    API --> DB[(SQLAlchemy DB\nSQLite local or Neon Postgres)]
    API --> RAG["RAG Engine\nOllama + ChromaDB\nlocal private runtime"]

    subgraph Loop["Active Learning Loop"]
        DB --> Dash[Streamlit\nAnalyst Review]
        Dash --> FB[Analyst Feedback]
        FB --> DB
        DB --> Retrain[Champion-Challenger\nRetraining]
        Retrain --> ML
    end

    subgraph Monitor["Observability"]
        CG -. metrics .-> Prom[Prometheus]
        API -. metrics .-> Prom
        Prom --> Grafana[Grafana\nDashboards + Alerts]
        Prom --> AM[Alertmanager\nPagerDuty/Slack/Email]
    end

    style Pipeline fill:#e1f5fe
    style Loop fill:#f3e5f5
    style Monitor fill:#e8f5e9
```

### End-to-End Data Flow

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '20px'}}}%%
sequenceDiagram
    participant P as Producer
    participant K as Kafka
    participant C as Spark Consumer
    participant R as Rules Engine
    participant M as XGBoost
    participant S as SAR Generator
    participant A as FastAPI
    participant DB as SQLAlchemy DB
    participant D as Dashboard

    P->>K: Generate dev/test TX events
    K->>C: Stream tunisian_transactions
    C->>C: Quality Gates (validate)
    C->>C: Stateful Aggregation (window per user)
    C->>R: Dynamic risk scoring
    R->>M: ML inference (champion model)
    M->>C: ML probability
    C->>S: Optional SAR draft (RAG + validation)
    S->>C: SAR report or deterministic fallback
    C->>A: POST alert + SHAP + optional SAR
    A->>DB: Persist alert, audit, feedback
    D->>A: Fetch review queue, stats, exports
    A->>D: Serve alert/review APIs
    D->>A: Submit analyst feedback
    A->>M: Trigger background retraining
    M->>DB: Update model registry
```

### Why These Technologies?

| Decision | Choice | Trade-off Rationale |
|----------|--------|---------------------|
| **Kafka over Redis Streams** | Confluent Kafka 7.7 | Durability, consumer groups, replay capability. Redis is faster but loses messages on restart. |
| **Spark over Benthos/Python workers** | PySpark 4.1.1 | Windowed stateful aggregation (5-min windows, per-user state) is native in Spark. Benthos lacks ML integration. Python workers don't handle watermarks. |
| **XGBoost before deep learning** | SparkXGBClassifier | Fast, interpretable tabular baseline suitable for limited labelled feedback. Graph, sequence, and anomaly models are still planned before any state-of-the-art claim. |
| **SQLAlchemy persistence** | SQLite local / PostgreSQL on Neon | Local development keeps a zero-dependency SQLite fallback. Hugging Face Spaces deployment uses Neon PostgreSQL through the same SQLAlchemy session layer. |
| **Ollama over Cloud LLMs** | Llama 3.1 local | PII never leaves the infrastructure. Cloud APIs violate Tunisian data residency requirements. |
| **ChromaDB over Pinecone** | ChromaDB local | Same data residency concern. Embeddings (all-MiniLM-L6-v2) run locally. |

---

## System Overview

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TB
    subgraph Ingestion["Layer 1: Ingestion (Bronze)"]
        P1[Producer]
        P2[Chaos Producer]
        K[Kafka Broker]
    end

    subgraph Processing["Layer 2: Processing (Silver)"]
        C[Spark Consumer]
        QG[Quality Gates]
        SA[Stateful Windowed Aggregation]
        RE[Dynamic Rules Engine]
    end

    subgraph Intelligence["Layer 3: Intelligence (Gold)"]
        ML[XGBoost ML]
        RS[Risk Scoring]
        AD["Alert Dispatch\nhigh risk + samples"]
    end

    subgraph Compliance["Layer 4: Compliance"]
        RAG["RAG Engine\nlocal private runtime"]
        SAR[SAR Generator + Validator]
        CTAF[CTAF Export]
    end

    subgraph Operations["Layer 5: Operations"]
        API[FastAPI Command Center\nHF Space verified]
        DB[(SQLAlchemy Persistence\nSQLite local or Neon deployed)]
        Dash[Streamlit Dashboard]
        Mon[Prometheus + Grafana]
        DLQ[Dead Letter Queue]
    end

    P1 --> K
    P2 --> K
    K --> C
    C --> QG
    QG --> SA
    SA --> RE
    RE --> RS
    RS --> ML
    ML --> AD
    AD --> RAG
    RAG --> SAR
    SAR --> AD
    AD --> API
    API --> DB
    SAR --> CTAF
    Dash --> API
    API --> Dash
    C -. metrics .-> Mon
    API -. metrics .-> Mon
    C --> DLQ
    DLQ --> API

    style Ingestion fill:#fff3e0
    style Processing fill:#e3f2fd
    style Intelligence fill:#e8f5e9
    style Compliance fill:#fce4ec
    style Operations fill:#f3e5f5
```

---

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Stream Ingestion | Apache Kafka (Confluent) | 7.7.0 | Durable, ordered message streaming |
| Stream Processing | PySpark Structured Streaming | 4.1.1 | Stateful windowed aggregation, watermarks |
| ML Model | XGBoost (SparkML) | 3.1.3 | Fraud classification with feature importance |
| Pipeline ML | Scikit-Learn + imbalanced-learn | 1.5.2 / 0.12.3 | Proper pipeline with SMOTE, no data leakage |
| Vector Store | ChromaDB | 0.5.0 | CTAF regulatory document retrieval |
| Embeddings | SentenceTransformers | 3.0.1 | all-MiniLM-L6-v2 semantic search |
| LLM | Ollama (Llama 3.1) | latest | Local SAR generation (data residency) |
| API | FastAPI | 0.115.0 | Command Center REST API |
| API Hosting | Hugging Face Spaces | Docker SDK | Verified free-tier deployment for the FastAPI service |
| Dashboard | Streamlit | 1.39.0 | Analyst operational UI |
| Monitoring | Prometheus + Grafana | 2.51 / 10.4 | Pipeline observability, alerting |
| Database | SQLAlchemy + SQLite/PostgreSQL | 2.0.35 / Neon verified | Feedback, alerts, model registry, audit logs, pKYC triggers |
| Tracing | OpenTelemetry | 1.27.0 | Distributed tracing |
| Secrets | HashiCorp Vault adapter | 2.3.0 | Enterprise secret management |

---

## Project Structure

```
├── k8s/                          # Kubernetes manifests (production deployment)
│   ├── namespace.yml             # Namespace definition
│   ├── kafka.yml                 # Kafka + Zookeeper StatefulSets
│   ├── consumer.yml              # Spark Consumer + PVCs
│   ├── api.yml                   # FastAPI deployment + Ingress
│   ├── ollama.yml                # Ollama (GPU) + ChromaDB
│   └── config.yml                # ConfigMap + Secrets
├── monitoring/                   # Production observability
│   ├── prometheus.yml            # Prometheus scrape config + alert rules
│   ├── alert_rules.yml           # 12 alert rules (latency, lag, error rate)
│   ├── alertmanager.yml          # Notification routing (email/Slack/PagerDuty)
│   ├── metrics_exporter.py       # Prometheus metrics for all pipeline components
│   ├── docker-compose.monitoring.yml  # Monitoring stack extension
│   └── grafana_dashboards/       # Pre-built Grafana dashboards
├── migrations/                   # Database schema migrations
│   ├── 0001_initial_schema.py    # Baseline: alerts, feedback, registry, audit, DLQ
│   ├── 0002_pii_anonymization.py # PII compliance columns + indexes
│   └── 0003_rules_engine.py      # Dynamic rules engine tables
├── dashboard/
│   ├── api.py                    # FastAPI Command Center (/api/v1/)
│   ├── dashboard.py              # Streamlit analyst dashboard
│   └── monitoring.py             # ForensicAnalyticEngine (latency, drift)
├── src/
│   ├── ml/
│   │   ├── train_pipeline.py     # ★ Proper ML pipeline (no data leakage)
│   │   └── train_model.py        # Spark-based champion-challenger retraining
│   ├── streaming/
│   │   ├── consumer.py           # Original Spark consumer
│   │   ├── consumer_stateful.py  # ★ Stateful consumer (mapGroupsWithState equivalent)
│   │   └── consumer_demo.py      # Lightweight dev consumer
│   ├── rag_engine/
│   │   ├── sar_generator.py      # ★ RAG + LLM validation + deterministic fallback
│   │   ├── sar_validator.py      # Pydantic schema validation for SAR output
│   │   └── vector_store.py       # ChromaDB CTAF regulation store
│   ├── shared/
│   │   ├── rules_engine.py       # ★ Dynamic rules (hot-reload, no hard-coded values)
│   │   ├── pii_masking.py        # GDPR/Tunisian law PII anonymization
│   │   ├── vault_client.py       # HashiCorp Vault adapter
│   │   ├── tracing.py            # OpenTelemetry distributed tracing
│   │   ├── schemas.py            # Pydantic + Spark schemas (SSoT)
│   │   ├── risk_config.py        # Compiled defaults (fallback for rules engine)
│   │   ├── quality_gates.py      # Data quality validation
│   │   ├── utils.py              # API helpers, DLQ, logging
│   │   └── logging_config.py     # Structured JSON logging
│   └── producer/
│       ├── producer.py           # Local transaction simulator for dev/test only
│       └── chaos_producer.py     # Chaos: delayed/malformed test transactions
├── tests/
│   ├── test_chaos_integration.py # ★ Kafka failure, checkpoint recovery, LLM fallback
│   ├── test_api.py               # FastAPI endpoint tests
│   ├── test_quality_gates.py     # PySpark quality gate tests
│   ├── test_schemas.py           # Schema validation tests
│   ├── test_risk_config.py       # Rules engine tests
│   ├── test_monitoring.py        # Monitoring engine tests
│   ├── test_producer.py          # Producer tests
│   └── test_utils.py             # Utility tests
├── scripts/
│   ├── bootstrap_imbalanced.py   # ★ Realistic 0.01% fraud rate data generation
│   ├── bootstrap_system.py       # Original bootstrap (for compatibility)
│   ├── migrate.py                # Database migration runner
│   ├── cost_estimate.py          # ★ Cloud infrastructure cost estimation
│   ├── audit_dependencies.py     # Dependency pin audit + CVE check
│   ├── load_test.py              # Async load testing (aiohttp)
│   └── run_tests.sh              # CI/CD test runner
├── notebooks/
│   └── 01_eda_fraud_detection.py # ★ EDA only (no training parameter derivation)
├── Makefile                      # ★ All operational commands in one file
├── requirements.txt              # ★ Fully pinned dependencies
├── docker-compose.yml            # Local dev stack
└── tests/                        # Public automated test suite
```

Files marked ★ are professional-grade additions that address the "amateur" markers.

Runtime outputs are intentionally not tracked. Spark parquet output, SQLite
databases, pytest caches, coverage reports, notebook HTML exports, model
registry artifacts, DLQ files, and local checkpoint/session handoff files are
ignored. Use `make clean` to remove generated artifacts from a local workspace.

---

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your tokens, database URL, Vault address, and PII salt
```

For local development, `DATABASE_URL=sqlite:///./data/feedback.db` is enough.
For Hugging Face Spaces + Neon, set the Space secrets to at least:

```bash
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST.neon.tech/DBNAME?sslmode=require
ADMIN_TOKEN=...
ANALYST_TOKEN=...
API_TOKEN=...
```

The FastAPI command center creates the core SQLAlchemy tables on startup:
`high_risk_alerts`, `feedback_labels`, `model_registry`, `audit_logs`, and
`pkyc_triggers`. Use managed migrations before treating the Neon database as a
regulated production store.

Deployment split:

- **Verified hosted slice:** Hugging Face Space running FastAPI on port `7860`, writing and reading alerts from Neon PostgreSQL.
- **Local/full-stack slice:** Docker Compose/Kubernetes-oriented Kafka, Spark consumer, Streamlit dashboard, Prometheus/Grafana, Ollama, and ChromaDB services.
- **Shared persistence contract:** SQLAlchemy session layer with SQLite fallback locally and Neon PostgreSQL in the hosted API.

### 2. One Command Full Setup

```bash
make setup        # Install pinned dependencies
make bootstrap    # Seed DB + train initial model
make migrate      # Apply database migrations
make prod         # Start full Docker Compose stack
```

### 3. Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8001/api/v1/ | Command Center REST API |
| API Docs | http://localhost:8001/docs | Swagger/OpenAPI |
| Dashboard | http://localhost:8501 | Streamlit analyst UI |
| Prometheus | http://localhost:9090 | Metrics & alert rules |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Alertmanager | http://localhost:9093 | Alert routing |
| ChromaDB | http://localhost:8000 | Vector store admin |

For the Hugging Face Docker Space, the API listens on port `7860`. The direct
Space host should expose `/health/`, `/docs`, and `/api/v1/...`. If the
Hugging Face web UI proxy returns a root-level `404`, test the direct
`*.hf.space` URL before debugging the API code.

### 4. Development Commands

```bash
make test          # Full test suite with coverage
make test-unit     # Unit tests only (no Spark/Java)
make chaos-test    # Failure/integration tests
make lint          # Flake8 + black + isort
make dev           # Local dev stack (no Docker)
```

### 5. Production Commands

```bash
make prod-monitor     # Full stack + Prometheus/Grafana
make monitor          # Monitoring stack only
make k8s-dry-run      # Validate K8s manifests
make k8s-apply        # Deploy to Kubernetes cluster
make cost-estimate    # Cloud cost projection
make audit-deps       # Dependency pin audit
```

---

## Risk Scoring Engine

### Dynamic Rules Engine

Risk rules are stored in the local rules database for development and can be updated by risk officers **without code deployment**. The command-center API persistence path now uses SQLAlchemy, with SQLite for local development and Neon PostgreSQL for the verified Hugging Face deployment:

```python
from src.shared.rules_engine import get_rules_engine

engine = get_rules_engine()
engine.update_rule("velocity", weight=0.35, threshold=4.0, changed_by="risk-officer-jane")
engine.force_refresh()  # Hot-reload across all consumers
```

All changes are audited in `rule_change_log` with before/after values and in a hash-chained JSONL audit log at `CHANGE_AUDIT_LOG` / `data/audit/change_audit.jsonl`.

### Risk Components

| Factor | Default Weight | Threshold | Description |
|--------|---------------|-----------|-------------|
| Velocity | 30% | >3 tx / 5min | Transaction frequency |
| Travel | 30% | >1 governorate | Impossible travel detection |
| High Value | 20% | >15000 TND | Enhanced-monitoring flag; not a cash-cap or structuring rule |
| Velocity Smurfing | configurable | count/aggregate window | Structuring review based on repeated sub-threshold transfers, not the repealed cash cap |
| D17 Wallet | 20% | >500 TND/day or >5 tx/day | D17 digital wallet limits (not Flouci) |
| Smurfing | +15% | Configurable range | Structuring pattern — validate against real transaction data |

---

## ML Pipeline: No Data Leakage

The **critical fix** in this version: all ML preprocessing is isolated within a Scikit-Learn Pipeline that is fit **only** on training data.

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    A[Full Dataset] --> B[Train/Test Split 80/20]
    B -->|Train Set| C[Preprocessor\nScaler + Imputer]
    B -->|Test Set| G[Hold Out — Touched Last]
    C --> D[SMOTE\nApplied to Train Only]
    D --> E[XGBoost Model\nFit on Preprocessed Train]
    E --> F[Evaluate\nPR-AUC, F1, Precision, Recall]
    G --> F

    subgraph Training["Training Pipeline"]
        B
        C
        D
        E
    end

    style Training fill:#e3f2fd
    style G fill:#fff3e0
    style F fill:#e8f5e9
```

See `src/ml/train_pipeline.py` for the full implementation.

### Evaluation Metrics (Proper for Imbalanced Data)

| Metric | Why We Use It | Why Accuracy Is Wrong |
|--------|--------------|----------------------|
| **F1 Score** | Balances precision and recall | Accuracy can be 99.99% by predicting "all legitimate" |
| **PR-AUC** | Focuses on the minority (fraud) class | ROC-AUC is inflated by the majority class |
| **Precision** | How many alerts are actually fraud? | |
| **Recall** | How many fraud cases did we catch? | |

---

## CTAF Compliance

- **SAR Generation**: RAG (Ollama + ChromaDB) with source-of-truth grounding, fact checks, **Pydantic schema validation**, hash-chained audit logging, and **deterministic fallback**
- **Filing Mandate**: 10-business-day deadline per CTAF/BCT
- **Human Gate**: SAR text is drafted for compliance review; it is not auto-submitted to CTAF
- **Export**: JSON CTAF export + structured SAR reports from recorded alerts and analyst feedback

### SAR Validation Flow

```
LLM Output → JSON Extraction → Pydantic Schema Validation → Pass? → Use LLM output
                                                         ↓
                                                    Fail? -> Deterministic Template
```

The system keeps SAR drafting grounded in stored facts and requires human approval before filing. The local ChromaDB seed documents are internal control text, not claimed official circulars; production deployments should load verified regulatory source documents.

---

## Security & Compliance

### PII Handling (GDPR + Tunisian Law 2004-63)

- User IDs are SHA-256 hashed before storage (`user_id_hashed`)
- Amounts can be masked for aggregate reporting
- Governorates can be generalized to regions for external exports
- k-anonymity checking before data release
- Data retention policies per data type (transactions: 5 years, SARs: 10 years)

### Secret Management

- **Development**: `.env` file (`.env.example` template)
- **Production**: HashiCorp Vault adapter (`src/shared/vault_client.py`)
- All secrets cached with 5-minute TTL
- Graceful fallback to environment variables if Vault is down

### RBAC

- **Admin**: Alert ingestion, model retraining, CTAF export, rule changes
- **Analyst**: Feedback submission, review queue, explainability
- Tokens are SHA-256 hashed before storage

---

## Monitoring & Observability

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_predictions_total` | Counter | Total predictions by alert type |
| `fraud_alerts_total` | Counter | Alerts by type and status |
| `fraud_ingestion_latency_seconds` | Histogram | Event-to-processing latency (P95/P99) |
| `kafka_consumer_lag` | Gauge | Consumer group lag in messages |
| `dlq_pending_count` | Gauge | Dead letter queue size |
| `model_f1_score` | Gauge | Current champion F1 |
| `feedback_precision` | Gauge | Analyst feedback precision |

### Alert Rules (12 rules)

| Alert | Severity | Condition | Impact |
|-------|----------|-----------|--------|
| KafkaConsumerLagHigh | Critical | Lag > 1000 for 5min | Detection latency |
| PipelineLatencyHigh | Warning | P95 > 2s for 5min | SLA violation |
| AlertErrorRateHigh | Critical | Error rate > 5% | Data loss risk |
| OllamaUnavailable | Warning | LLM down for 2min | SAR fallback active |
| ModelF1Degradation | Critical | F1 < 0.70 for 1h | Model retrain needed |
| FraudRateSpike | Critical | 3x hourly average | Possible attack |

See `monitoring/alert_rules.yml` for all rules.

---

## Infrastructure Cost

```bash
python scripts/cost_estimate.py --cloud all --tx-per-day 1000000
```

### Estimated Monthly Cost Scenario (1M tx/day, 0.01% fraud)

| Provider | Monthly | Per Transaction | Notes |
|----------|---------|----------------|-------|
| **AWS** (us-east-1) | ~$2,800 | $0.000093 | GPU instance is 35% of cost |
| **GCP** (us-central1) | ~$2,700 | $0.000090 | L4 GPU slightly cheaper |
| **Azure** (eastus) | ~$2,950 | $0.000098 | T4 GPU premium |
| **EU-Nearshore** (OVH/Hetzner) | ~$1,800–$2,200 | $0.000060–$0.000073 | Lower cost, closer to Tunisia, better data residency alignment |

**Cost optimization**: Running Ollama on CPU (no GPU) reduces cost by ~$800/month with 2-3x latency increase. For strict Tunisian data-residency compliance, consider EU-nearshore providers (OVH, Hetzner) or local providers to minimize cross-border data transfer.

See `scripts/cost_estimate.py` for full breakdown.

---

## Kubernetes Deployment

The system ships with production-ready K8s manifests:

- **StatefulSets** for Kafka and Zookeeper with persistent volumes
- **Deployments** with resource requests/limits for all services
- **Single API replica by default** for local SQLite; use PostgreSQL/Neon before scaling the command-center API horizontally
- **Ingress** with rate limiting (nginx annotations)
- **PVCs** for all persistent data (180GB total)
- **ConfigMaps** for hot-reloadable configuration
- **Secrets** for token management

```bash
make k8s-dry-run   # Validate manifests
make k8s-apply     # Deploy to cluster
kubectl get pods -n amastan  # Monitor
```

---

## License

See [LICENSE](LICENSE) for details.

---

## Professional Notes

This system was built with production requirements in mind:

1. **Every component handles failure**: Kafka disconnects, API downtime, LLM hallucination, Spark checkpoint corruption
2. **Data leakage is prevented**: The ML pipeline uses proper Scikit-Learn Pipelines with train/test isolation
3. **No hard-coded business logic**: Risk rules are database-driven and hot-reloadable
4. **Regulatory compliance**: SAR generation has factual grounding, deterministic fallback, and human approval before filing
5. **Observability is first-class**: Prometheus metrics, Grafana dashboards, and 12 alert rules
6. **Reproducibility**: Dependencies are version-pinned, migrations are versioned, and costs are estimated
7. **Data privacy**: PII is hashed, k-anonymity is checked, and retention policies are enforced

The architecture prioritizes **reliability over cleverness**. A fraud detection system that silently drops 1% of alerts due to unhandled errors is worse than one that uses simpler technology but handles every failure mode.
