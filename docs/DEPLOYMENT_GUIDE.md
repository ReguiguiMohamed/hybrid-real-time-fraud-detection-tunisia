# Deployment Guide

## Architecture Split

The system has two deployment slices:

| Slice | Runtime | Components | Status |
|-------|---------|------------|--------|
| **Verified Hosted** | Hugging Face Space + Neon PostgreSQL | FastAPI command-center API | Verified & running |
| **Local Full-Stack** | Docker Compose / Kubernetes | Kafka, Spark, Ollama, ChromaDB, Streamlit, Prometheus | Scaffolding |

## Verified Hosted Deployment (Hugging Face + Neon)

### Prerequisites

- Hugging Face account with Docker SDK space
- Neon PostgreSQL database (free tier sufficient)
- API tokens generated for admin, analyst, and metrics

### Setup

1. **Neon PostgreSQL**: Create a project and copy the connection string.

2. **Hugging Face Space**:
   - Create a new Docker Space
   - Set the `Dockerfile` as the build target (port 7860)
   - Configure secrets:

   ```
   ADMIN_TOKEN=<your-admin-token>
   ANALYST_TOKEN=<your-analyst-token>
   API_TOKEN=<your-admin-token>
   METRICS_TOKEN=<your-metrics-token>  # optional, for Grafana Cloud
   DATABASE_URL=postgresql+psycopg2://user:pass@host.neon.tech/db?sslmode=require
   ```

3. **Verify**:
   - Check `/health/` returns `{"status": "healthy"}`
   - Check `/docs` loads Swagger
   - Test `POST /api/v1/alerts/add/` with admin token
   - Test `GET /api/v1/alerts/high-risk/` shows the inserted alert

### Verified Path

```
FastAPI on Hugging Face Spaces -> authenticated /api/v1 alert endpoint -> SQLAlchemy -> Neon PostgreSQL -> API readback
```

## Local Full-Stack Deployment (Docker Compose)

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM for Ollama + Spark

### Quick Start

```bash
make setup        # Install dependencies
make bootstrap    # Seed DB + train model
make prod         # Start full Docker Compose stack
```

### Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:8001 |
| Dashboard | http://localhost:8501 |
| ChromaDB | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |
| Alertmanager | http://localhost:9093 |

### Configuration

Copy `.env.example` to `.env` and edit:
- `DATABASE_URL=sqlite:///./data/feedback.db` for local SQLite
- `ADMIN_TOKEN` and `ANALYST_TOKEN` for authentication
- Kafka, Ollama, and monitoring settings as needed

## Kubernetes (Target Architecture)

Manifests in `k8s/` are target architecture scaffolding. Deploy with:

```bash
make k8s-dry-run   # Validate
make k8s-apply     # Deploy
```

Requires a provisioned cluster. Not currently a verified runtime.

## Monitoring

### Grafana Cloud (Hosted API)

1. Set `METRICS_TOKEN` in Space secrets
2. Configure Grafana Cloud Metrics Endpoint to scrape `https://<space>.hf.space/metrics`
3. Import `monitoring/grafana_dashboards/hosted_api_persistence.json`

### Local (Docker)

```bash
make prod-monitor   # Full stack + Prometheus + Grafana
make monitor        # Monitoring stack only
```

## Database

| Environment | Database | Connection |
|-------------|----------|------------|
| Local dev | SQLite | `sqlite:///./data/feedback.db` |
| Hosted API | Neon PostgreSQL | Via `DATABASE_URL` secret |
| Production target | PostgreSQL with TDE | Managed migration |
