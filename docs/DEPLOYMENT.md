# Deployment

The demo uses a Hugging Face Docker Space and Neon PostgreSQL.

## Required Secrets

Set these in the Space settings:

```text
ADMIN_TOKEN=<random value>
ANALYST_TOKEN=<random value>
METRICS_TOKEN=<random value>
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

`ADMIN_TOKEN` can ingest alerts and export cases. `ANALYST_TOKEN` can read the
queue and submit feedback.

## Deploy

Push to `main`.

The GitHub Actions deployment copies the runtime files to the Hugging Face
Space. The Space builds [`Dockerfile`](../Dockerfile) and serves the API on port
`7860`.

## Check

```powershell
Invoke-RestMethod https://<space>.hf.space/health/
```

Then check:

- `/docs`
- an authenticated `POST /api/v1/alerts/add/`
- `GET /api/v1/alerts/review-queue/`
- the inserted row in Neon

## Grafana Cloud

Grafana Cloud scrapes:

```text
https://<space>.hf.space/metrics
```

Use bearer authentication with `METRICS_TOKEN`. Import
[`grafana-dashboard.json`](grafana-dashboard.json).

Useful checks:

```promql
amastan_api_info
amastan_db_metrics_scrape_success
sum(rate(amastan_api_requests_total{status_code=~"5.."}[5m]))
```

## Limits

The application creates its prototype tables with SQLAlchemy on startup. That
is enough for the demo. It is not a production migration strategy.

Before using another database:

- rehearse schema changes;
- configure backups;
- define retention;
- review connection limits;
- rotate all tokens.
