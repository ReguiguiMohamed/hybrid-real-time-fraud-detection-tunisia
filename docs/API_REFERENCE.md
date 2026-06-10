# Amastan Fraud Shield Guard — API Reference

Base URL: `/api/v1/`

## Authentication

All endpoints require a `Bearer` token in the `Authorization` header.

| Role | Token Env Variable | Access |
|------|--------------------|--------|
| Admin | `ADMIN_TOKEN` / `API_TOKEN` | All endpoints |
| Analyst | `ANALYST_TOKEN` | Read + feedback write |

Tokens are SHA-256 hashed before comparison.

## Endpoints

### Health & Discovery

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | None | Root status with route links |
| GET | `/health/` | None | Health check |
| GET | `/docs` | None | Swagger UI |
| GET | `/openapi.json` | None | OpenAPI 3.1 spec |
| GET | `/metrics` | Metrics token | Prometheus metrics endpoint |

### Authentication

#### `GET /api/v1/auth/whoami`

Returns the authenticated user's role and available links.

**Response:**
```json
{
  "user_id": "analyst-01",
  "role": "ANALYST",
  "authenticated": true,
  "_links": { ... }
}
```

### Alerts

#### `POST /api/v1/alerts/add/` [Admin]

Ingest a high-risk alert from the streaming pipeline.

**Request body:** `TransactionAlert` — transaction_id, user_id, amount_tnd, governorate, payment_method, ml_probability, timestamp, optional fields.

#### `GET /api/v1/alerts/high-risk/`

Fetch high-risk alerts (ml_probability > 0.85). Query params: `limit` (default 50), `branch_id`.

#### `GET /api/v1/alerts/review-queue/`

Fetch the analyst review queue. Query params: `limit` (default 100), `alert_type`, `branch_id`.

#### `GET /api/v1/alerts/{transaction_id}/explain`

Explain risk factors using SHAP feature importance or champion model feature importance.

#### `GET /api/v1/alerts/{transaction_id}/export`

Export a single alert for compliance filing, including analyst review if available.

#### `GET /api/v1/alerts/ctaf-export` [Admin]

Export confirmed fraud alerts in CTAF-reporting format. Query params: `days` (default 7), `branch_id`.

### Feedback

#### `POST /api/v1/feedback/`

Submit analyst feedback on a fraud prediction.

**Request body:**
```json
{
  "transaction_id": "TXN_001",
  "analyst_label": "Confirmed Fraud",
  "analyst_comment": "Pattern matches known smurfing behavior",
  "branch_id": "Tunis-GNC"
}
```
`analyst_label` must be `"Confirmed Fraud"` or `"False Positive"`.

#### `POST /api/v1/feedback/batch/`

Submit multiple feedback entries at once.

```json
{
  "feedback_items": [
    { "transaction_id": "TXN_001", "analyst_label": "Confirmed Fraud" },
    { "transaction_id": "TXN_002", "analyst_label": "False Positive" }
  ]
}
```

### Statistics & Compliance

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/stats/` | System statistics with optional `branch_id` filter |
| GET | `/api/v1/compliance/kpis/` | CTAF compliance KPIs (SAR timeliness, sanctions, pKYC) |
| GET | `/api/v1/branches/` | List distinct branch IDs with alerts |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/monitoring/model-performance/` | Precision, recall, F1 from human feedback |
| GET | `/api/v1/metrics/performance` | Model performance metrics and drift indicators |
| GET | `/api/v1/metrics/feedback` | Analyst feedback distribution |
| GET | `/api/v1/metrics/threshold-analysis` | ML threshold trade-off analysis |
| GET | `/api/v1/metrics/drift` | Feature drift and retraining assessment |
| GET | `/api/v1/metrics/system-overview` | Aggregated system metrics |

### Model Management

#### `POST /api/v1/retrain-model/` [Admin]

Queue champion/challenger retraining using accumulated feedback. Returns `202`
with a job ID. PostgreSQL deployments disable this endpoint by default because
the hosted API image is not a Spark training runtime.

#### `GET /api/v1/retrain-model/status/{job_id}` [Admin]

Return the observable state of a queued job: `queued`, `running`, `no_change`,
`promoted`, or `failed`.

### Legacy Endpoints (Deprecated)

Unversioned path aliases exist at `/feedback/`, `/alerts/add/`, `/stats/`, etc. These delegate to the v1 handlers and will be removed in a future release.

## Error Responses

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 401 | Unauthorized (missing/invalid token) |
| 403 | Forbidden (valid token, wrong scope) |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

## HATEOAS Links

Responses from key endpoints include a `_links` object with related resource URLs:

| Endpoint | Links |
|----------|-------|
| whoami | self, stats, review_queue, branches, compliance_kpis, model_performance |
| stats | self, review_queue, compliance_kpis, model_performance, ctaf_export, branches |
| feedback | feedback, explain, stats |
| feedback/batch | feedback, feedback_batch, stats |
| ctaf-export | self, stats, review_queue |
