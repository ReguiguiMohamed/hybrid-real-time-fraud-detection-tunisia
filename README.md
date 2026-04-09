# Amastan - Fraud Shield Guard: Hybrid Streaming & RAG Architecture

A real-time fraud mitigation engine for Tunisian digital payments. Uses **Kafka + Spark Structured Streaming** for millisecond detection and **RAG (Ollama/ChromaDB)** for automated CTAF-compliant reporting.

The topic came into fruition ever since the introduction of incentives on 'cashless' transactions in Tunisia during January-February 2026, a period marked by the highest ever recorded liquidity rate in the country's history amid deepening inflation rates and economic uncertainty.

---
## Project Architecture and Data Flow 
<img width="1514" height="651" alt="ChatGPT Image Apr 9, 2026, 04_04_45 PM" src="https://github.com/user-attachments/assets/b2db5032-e61e-42eb-bc78-8109f295a111" />

1. **Bronze Layer**: Producer generates Tunisian transactions → Kafka topic `tunisian_transactions`
2. **Silver Layer**: Spark Structured Streaming consumes, applies quality gates, enriches with windowed analytics
3. **Gold Layer**: XGBoost ML scoring + weighted risk engine → alerts sent to Command Center API
4. **RAG Layer**: High-risk alerts trigger SAR generation via Ollama + ChromaDB regulatory context
5. **Feedback Loop**: Analyst feedback → model retraining → champion-challenger promotion

---

## January-February 2026 Economic Context

- **Liquidity Crisis**: Peak monetary expansion following unprecedented fiscal stimulus measures
- **Inflation Surge**: Double-digit inflation rates destabilizing purchasing power
- **Digital Payment Boom**: Government incentives for cashless transactions to digitize economy
- **Fraud Vulnerability**: Rapid digital adoption creating new attack vectors for financial crime
- **Regulatory Pressure**: CTAF mandates stricter AML/CFT compliance amid economic instability


---

## Project Structure

```
├── dashboard/
│   ├── api.py              # FastAPI Command Center (versioned /api/v1/)
│   ├── dashboard.py        # Streamlit operational dashboard
│   └── monitoring.py       # ForensicAnalyticEngine (latency, drift, threshold)
├── src/
│   ├── ml/
│   │   ├── train_model.py      # Champion-challenger retraining pipeline
│   │   └── train_model_mock.py # Mock trainer for development
│   ├── producer/
│   │   ├── producer.py         # Transaction generator (Faker → Kafka)
│   │   └── chaos_producer.py   # Chaos engineering: delayed/malformed transactions
│   ├── rag_engine/
│   │   ├── sar_generator.py    # SAR report generation (Ollama + retry backoff)
│   │   └── vector_store.py     # ChromaDB CTAF regulation store
│   ├── shared/
│   │   ├── schemas.py          # Pydantic Transaction model + Spark schema
│   │   ├── risk_config.py      # Risk weights, D17 thresholds, CBDC zones
│   │   ├── quality_gates.py    # Data quality validation (24 governorates)
│   │   ├── utils.py            # API helpers, DLQ, SQLite connections
│   │   └── logging_config.py   # Structured JSON logging
│   └── streaming/
│       ├── consumer.py         # Spark Structured Streaming fraud processor
│       └── consumer_demo.py    # Lightweight Kafka consumer for development
├── tests/
│   ├── conftest.py             # Shared fixtures (DB, API client, samples)
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_schemas.py         # Pydantic/Spark schema tests
│   ├── test_producer.py        # Transaction generator tests
│   ├── test_quality_gates.py   # Data quality gate tests (PySpark)
│   ├── test_risk_config.py     # Risk configuration tests
│   ├── test_utils.py           # Utility function tests
│   └── test_monitoring.py      # Monitoring engine tests
├── scripts/
│   ├── bootstrap_system.py     # Initial model + DB setup
│   ├── load_test.py            # Async load testing (aiohttp)
│   ├── run_tests.sh            # CI/CD test runner
│   └── update_pydantic_models.py
├── notebooks/
│   └── 01_eda_fraud_detection.py  # Exploratory data analysis
├── models/
│   ├── model_card.md           # Model documentation
│   └── fraud_xgb_v1/           # Trained model artifacts
├── docker-compose.yml          # Full stack orchestration (with health checks)
├── Dockerfile.api              # FastAPI service
├── Dockerfile.consumer         # Spark consumer service
├── Dockerfile.producer         # Transaction producer service
├── Dockerfile.dashboard        # Streamlit dashboard service
├── Dockerfile.ml               # ML training service
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── setup.py                    # Package configuration
└── test_end_to_end.py          # End-to-end integration test
```

---

## Quick Start

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your tokens and configuration
```

### 2. Docker Compose (Full Stack)

```bash
docker compose up --build
```

This starts: Zookeeper, Kafka, ChromaDB, Ollama, API, Dashboard, Producer, Consumer.

All services include health checks and proper dependency ordering.

### 3. Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8001/api/v1/ | Command Center REST API |
| API Docs | http://localhost:8001/docs | Swagger/OpenAPI documentation |
| Dashboard | http://localhost:8501 | Streamlit operational dashboard |
| Health | http://localhost:8001/health/ | API health check |
| ChromaDB | http://localhost:8000 | Vector store admin |

### 4. Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Bootstrap the system (creates DB + initial model)
python scripts/bootstrap_system.py

# Start the API
uvicorn dashboard.api:app --host 0.0.0.0 --port 8001

# In another terminal, start the producer
PYTHONPATH=src python src/producer/producer.py --rate 2

# Start the dashboard
streamlit run dashboard/dashboard.py
```

### 5. Run Tests

```bash
# Quick unit tests (no Spark required)
bash scripts/run_tests.sh --unit-only --verbose

# Full test suite (requires PySpark + Java)
bash scripts/run_tests.sh --coverage --verbose
```

---

## API Endpoints (v1)

All business endpoints are under `/api/v1/`. Legacy (unversioned) paths are supported for backward compatibility.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/auth/whoami` | Analyst/Admin | Current user info |
| GET | `/api/v1/alerts/review-queue/` | Analyst/Admin | Alert review queue |
| GET | `/api/v1/alerts/high-risk/` | Analyst/Admin | High-risk alerts only |
| POST | `/api/v1/alerts/add/` | Admin | Ingest alert from pipeline |
| GET | `/api/v1/alerts/{id}/explain` | Analyst/Admin | Explainability factors |
| GET | `/api/v1/alerts/{id}/export` | Analyst/Admin | Export single alert |
| GET | `/api/v1/alerts/ctaf-export` | Admin | CTAF compliance export |
| POST | `/api/v1/feedback/` | Analyst/Admin | Submit analyst feedback |
| GET | `/api/v1/stats/` | Analyst/Admin | System statistics |
| GET | `/api/v1/branches/` | Analyst/Admin | List active branches |
| GET | `/api/v1/monitoring/model-performance/` | Analyst/Admin | Sampling-aware metrics |
| GET | `/api/v1/metrics/system-overview` | Analyst/Admin | Full system overview |
| POST | `/api/v1/retrain-model/` | Admin | Trigger model retraining |
| GET | `/health/` | Public | Health check |

---

## Risk Scoring Engine

### Weighted Risk Components

| Factor | Weight | Description |
|--------|--------|-------------|
| Velocity | 30% | Transaction frequency in 5-min sliding window |
| Travel | 30% | Distinct governorates (impossible travel detection) |
| High Value | 20% | Amount > 5000 TND threshold |
| D17 Limit | 20% | Flouci/e-wallet specific limits (smurfing range 1400-1500 TND) |

### ML Pipeline

- **Algorithm**: XGBoost (SparkXGBClassifier)
- **Features**: v_count, g_dist, avg_amount, is_smurfing, high_velocity_flag
- **Retraining**: Champion-challenger with F1 improvement threshold (default 2%)
- **Active Learning**: Uncertainty zone sampling (0.4-0.6 probability) + random low-risk sampling

---

## CTAF Compliance

- **SAR Generation**: Automated via RAG (Ollama + ChromaDB CTAF regulatory context)
- **Filing Mandate**: 10-business-day filing deadline per CTAF/BCT requirements
- **Regulations Indexed**: Circular 2024-03, 2024-05, 2025-01; BCT Note 2025-02; AML Guidelines 2025
- **Export**: JSON CTAF export endpoint for confirmed fraud cases

---

## Security Features

- **RBAC**: Admin/Analyst role-based access control via bearer tokens
- **Rate Limiting**: Configurable per-IP rate limiting (default 60 req/min)
- **CORS**: Configurable allowed origins
- **Input Validation**: Pydantic field validators on all API inputs
- **Audit Logging**: All feedback, model promotions, and admin actions logged
- **Dead Letter Queue**: Failed alerts stored in SQLite DLQ with automatic retry

### Production Hardening Recommendations

- Store tokens in HashiCorp Vault or AWS Secrets Manager
- Implement certificate-based inter-service authentication
- Add DDoS protection (e.g., Cloudflare, AWS WAF)
- Enable TLS for all inter-service communication
- Configure network policies in Kubernetes deployment

---

## Monitoring & Observability

- **Structured JSON Logging**: All services emit structured logs for aggregation
- **Inference Latency**: P95/P99 tracking in ForensicAnalyticEngine
- **Data Drift Detection**: KS-test based distribution monitoring
- **Threshold Analysis**: Automated optimal threshold recommendation from feedback
- **Ingestion Latency**: End-to-end latency from event timestamp to API storage

---

## License

See [LICENSE](LICENSE) for details.
