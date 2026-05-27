# Grafana Cloud Free-Plan Setup

Checked against Grafana Cloud documentation on 2026-05-27.

This project should be monitored first through the verified hosted slice:

`FastAPI on Hugging Face Spaces -> authenticated /api/v1 alert endpoint -> SQLAlchemy -> Neon PostgreSQL -> API readback`

The Kafka, Spark, Ollama, ChromaDB, Streamlit, Prometheus, and Kubernetes pieces
remain local/target architecture until they are deployed and verified separately.

## What Is Ready in This Repo

- `GET /metrics` on the FastAPI app exports Prometheus-compatible aggregate metrics.
- `METRICS_TOKEN` protects `/metrics` with Bearer authentication. On PostgreSQL/Neon deployments, `/metrics` fails closed when this token is not set.
- `monitoring/grafana_dashboards/hosted_api_persistence.json` is the Grafana Cloud dashboard for the hosted API + Neon persistence slice.
- `monitoring/prometheus.grafana-cloud.example.yml` is only an optional collector-based fallback. The recommended free-plan setup is Grafana Cloud Metrics Endpoint integration because it can scrape the public Hugging Face Space directly.

## Hugging Face Space Secret

Add this Space secret before configuring Grafana:

```text
METRICS_TOKEN=<long-random-token>
```

After the Space redeploys, verify locally:

```powershell
Invoke-WebRequest `
  -Uri "https://<your-space>.hf.space/metrics" `
  -Headers @{ Authorization = "Bearer <long-random-token>" }
```

The response should include metrics such as:

```text
amastan_api_info
amastan_api_requests_total
amastan_db_alerts_total
amastan_db_metrics_scrape_success
```

## Grafana Cloud UI Steps

### 1. Create the Free Cloud Stack

1. Go to `https://grafana.com/` and sign in or create a free account.
2. Create or open your Grafana Cloud stack.
3. Open the stack's Grafana instance.

Grafana Cloud free access includes managed Grafana plus hosted metrics capacity. Use the hosted stack; do not create a local Grafana instance for the portfolio deployment.

### 2. Add the Hugging Face Metrics Endpoint

1. In the Grafana Cloud stack, open **Connections** from the left navigation.
2. Select **Add new connection** if shown.
3. Search for **Metrics Endpoint**.
4. Open the **Metrics Endpoint** integration.
5. Create a scrape job:
   - Job name: `amastan-hf-api`
   - URL: `https://<your-space>.hf.space/metrics`
   - Scrape interval: `1 minute`
   - Authentication: `Bearer`
   - Token: paste the `METRICS_TOKEN` value only, without the `Bearer ` prefix.
6. Click **Test connection**.
7. Click **Save scrape job**.
8. If Grafana offers to install the integration dashboard, install it. It is useful for scrape health, not for fraud-specific KPIs.

The Metrics Endpoint integration requires an HTTPS, publicly reachable, authenticated Prometheus/OpenMetrics endpoint. Hugging Face Spaces satisfies this when the direct `*.hf.space` URL is used.

### 3. Import the Project Dashboard

1. Open **Dashboards** in Grafana.
2. Click **New**.
3. Choose **Import**.
4. Upload `monitoring/grafana_dashboards/hosted_api_persistence.json`.
5. When prompted for the Prometheus data source, choose the Grafana Cloud Metrics/Prometheus data source for your stack.
6. Click **Import**.

The dashboard should show the hosted API state, request rates, p95 latency, persisted alerts by type, review queue count, feedback count, and SQLAlchemy/Neon readback health.

## Evidence To Keep In The Repo

Keep these artifacts in GitHub so the result is easy to review later:

- The dashboard JSON export from Grafana Cloud
- The repo copy in `monitoring/grafana_dashboards/hosted_api_persistence.json`
- The combined dashboard + Explore screenshot in `docs/grafana.png`
- The verification screenshot in `resultscreenshot.png`
- The deployed runtime notes in the README, centered on the hosted path

That combination is stronger than only describing the stack in prose because it shows the live dashboard, the captured config, and the deployment proof together.

### 4. Validate Data in Explore

Open **Explore**, select the Grafana Cloud Prometheus data source, and run:

```promql
amastan_api_info
```

Then run:

```promql
amastan_db_metrics_scrape_success
```

Expected value is `1`. If it is `0`, Grafana can reach `/metrics` but the API failed to query the SQLAlchemy database during scrape.

### 5. Recommended Alert Rules

Create these in **Alerting -> Alert rules** after data is flowing:

```promql
absent(amastan_api_info{runtime="huggingface-space"})
```

Use this for Space/API scrape absence.

```promql
amastan_db_metrics_scrape_success == 0
```

Use this for SQLAlchemy/Neon readback failure.

```promql
sum(rate(amastan_api_requests_total{status_code=~"5.."}[5m])) > 0
```

Use this for API server errors.

## Optional Prometheus Remote Write

Use `monitoring/prometheus.grafana-cloud.example.yml` only if you decide to run your own Prometheus collector. Grafana Cloud's official Prometheus path uses `remote_write` with:

- URL: Grafana Cloud Metrics `/api/prom/push` endpoint
- username: Metrics instance ID
- password: Cloud Access Policy token

For the current Hugging Face deployment, the Metrics Endpoint integration is simpler and avoids another always-on service.

## Kubernetes Is Not the Current Path

Do not present Kubernetes as deployed for this project yet. When a real cluster exists, use Grafana Cloud's Kubernetes Monitoring path:

1. Open **Connections**.
2. Search **Kubernetes Monitoring**.
3. Use the Helm-chart flow.
4. Grafana Cloud deploys Grafana Alloy through the Kubernetes Monitoring Helm chart.

That is a later infrastructure workstream. It should be documented separately from the current Hugging Face + Neon verification.

## References

- Grafana Cloud overview: https://grafana.com/docs/grafana/latest/introduction/grafana-cloud/
- Metrics Endpoint integration: https://grafana.com/docs/grafana-cloud/monitor-infrastructure/integrations/integration-reference/integration-metrics-endpoint/
- Prometheus remote write to Grafana Cloud: https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/
- Access policies and tokens: https://grafana.com/docs/grafana-cloud/security-and-account-management/authentication-and-permissions/access-policies/create-access-policies/
- Kubernetes Monitoring configuration: https://grafana.com/docs/grafana-cloud/monitor-infrastructure/kubernetes-monitoring/configuration/
