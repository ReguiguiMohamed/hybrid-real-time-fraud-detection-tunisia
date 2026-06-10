# Deployment: Choose Your Path

## Option A — Hosted API (Hugging Face Space + Neon)

**Best for**: A hosted prototype API serving dashboards, analysts, and compliance tools.

| What | Details |
|------|---------|
| Components | FastAPI command-center API only |
| Database | Neon PostgreSQL (free tier works) |
| Auth | Bearer tokens (admin, analyst, metrics) |
| Persistence | Full — alerts, feedback, KPIs survive restarts |
| Cost | ~$0 (HF Space free tier + Neon free tier) |
| Limits | 512MB RAM, cold starts after inactivity |
| Retraining | Disabled by default on PostgreSQL; requires a separate Spark-capable runtime |

**Setup**: see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#verified-hosted-deployment-hugging-face--neon)

---

## Option B — Local Full-Stack (Docker Compose)

**Best for**: Development, demonstration, or when you need the full pipeline.

| What | Details |
|------|---------|
| Components | API, Streamlit dashboard, Kafka, Spark, Ollama, ChromaDB, Prometheus/Grafana |
| Database | SQLite by default |
| Auth | Same bearer tokens (set in `.env`) |
| Persistence | SQLite file in `./data/` |
| Resources | 8GB+ RAM recommended (Ollama + Spark) |

**Setup**: `make setup && make bootstrap && make prod`

---

## Can't decide?

- **Only need the REST API for integration?** → Option A
- **Building/tuning the ML pipeline?** → Option B (has Spark, Ollama, backtesting)
- **Working on the Streamlit dashboard?** → Option B
- **Doing compliance or CTAF reporting?** → Option A (persistent Neon DB + always-on)

---

## What about Kubernetes?

Manifests in `k8s/` are **architecture scaffolding only**. Not a verified runtime. Do not deploy.
