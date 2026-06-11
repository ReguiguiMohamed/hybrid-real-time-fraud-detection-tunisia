# API Reference

Base path: `/api/v1`

Swagger: `/docs`

OpenAPI: [`openapi.json`](openapi.json)

## Authentication

Send a bearer token:

```http
Authorization: Bearer <token>
```

`ADMIN_TOKEN` has full access. `ANALYST_TOKEN` can read alerts and submit
feedback. `/health/` and `/docs` are public. `/metrics` uses `METRICS_TOKEN`.

## Alerts

### `POST /api/v1/alerts/add/`

Admin only.

```json
{
  "transaction_id": "TXN_001",
  "user_id": "USER_12",
  "amount_tnd": 3200,
  "governorate": "Tunis",
  "payment_method": "card",
  "timestamp": "2026-06-11T10:00:00Z",
  "ml_probability": 0.94
}
```

### `GET /api/v1/alerts/high-risk/`

Returns high-risk alerts.

### `GET /api/v1/alerts/review-queue/`

Returns alerts waiting for analyst review.

Optional query parameters:

- `limit`
- `alert_type`
- `branch_id`

### `GET /api/v1/alerts/{transaction_id}/explain`

Returns stored SHAP values or model feature importance.

### `GET /api/v1/alerts/{transaction_id}/export`

Exports one reviewed case.

### `GET /api/v1/alerts/ctaf-export`

Admin only. Exports confirmed cases for a selected time window.

## Feedback

### `POST /api/v1/feedback/`

```json
{
  "transaction_id": "TXN_001",
  "analyst_label": "Confirmed Fraud",
  "analyst_comment": "Reviewed against account history"
}
```

`analyst_label` must be `Confirmed Fraud` or `False Positive`.

### `POST /api/v1/feedback/batch/`

Stores several feedback items in one request.

## Reporting

| Method | Path |
|---|---|
| `GET` | `/api/v1/stats/` |
| `GET` | `/api/v1/compliance/kpis/` |
| `GET` | `/api/v1/branches/` |
| `GET` | `/api/v1/monitoring/model-performance/` |
| `GET` | `/api/v1/metrics/performance` |
| `GET` | `/api/v1/metrics/feedback` |
| `GET` | `/api/v1/metrics/threshold-analysis` |
| `GET` | `/api/v1/metrics/drift` |
| `GET` | `/api/v1/metrics/system-overview` |
| `GET` | `/api/v1/model/training-summary` |

## Status Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `401` | Missing or invalid token |
| `403` | Wrong role |
| `404` | Not found |
| `422` | Invalid request |
| `429` | Rate limit reached |
| `500` | Server error |
