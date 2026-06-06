# Operational Runbook

## Health Checks

```bash
# API health
curl https://<host>/health/

# Prometheus metrics
curl -H "Authorization: Bearer $METRICS_TOKEN" https://<host>/metrics

# Database connectivity (via stats endpoint)
curl -H "Authorization: Bearer $ADMIN_TOKEN" https://<host>/api/v1/stats/
```

## Common Tasks

### View Review Queue
```bash
curl -H "Authorization: Bearer $ANALYST_TOKEN" \
  https://<host>/api/v1/alerts/review-queue/?limit=50
```

### Submit Analyst Feedback
```bash
curl -X POST -H "Authorization: Bearer $ANALYST_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"TXN_001","analyst_label":"Confirmed Fraud","analyst_comment":"Verified suspicious"}' \
  https://<host>/api/v1/feedback/
```

### Trigger Model Retraining
```bash
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://<host>/api/v1/retrain-model/
```

### Export CTAF Report
```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://<host>/api/v1/alerts/ctaf-export?days=30
```

## Alert Response Guide

| Alert | Severity | Response |
|-------|----------|----------|
| KafkaConsumerLagHigh | Critical | Check consumer group status, restart consumer if needed |
| PipelineLatencyHigh | Warning | Check Spark batch duration, scale executors |
| AlertErrorRateHigh | Critical | Inspect API error logs, check database connectivity |
| OllamaUnavailable | Warning | SAR fallback activated — check Ollama service |
| ModelF1Degradation | Critical | Trigger retraining, check feature drift |
| FraudRateSpike | Critical | Verify with manual review, possible attack in progress |

## Recovery Procedures

### Database Connection Loss

1. Check `DATABASE_URL` environment variable
2. Neon: connection pooling recycles SSL connections — pool_pre_ping handles this
3. SQLite: check disk space and file permissions

### DLQ Processing

The dead letter queue stores alerts that failed initial ingestion. A background worker retries on a configurable interval (`DLQ_RETRY_INTERVAL_SECONDS`, default 60).

```bash
# Manual retry (if worker is not running)
python -c "from shared.utils import retry_failed_alerts; retry_failed_alerts(max_attempts=3)"
```

### Model Rollback

If a retrained model degrades F1, the previous champion remains in `model_registry`. To manually roll back:

```sql
UPDATE model_registry SET is_champion = 0 WHERE is_champion = 1;
UPDATE model_registry SET is_champion = 1 WHERE version_id = 'v1_original';
```

## Maintenance

### Daily
- Review Grafana dashboard for alert rule violations
- Check DLQ size (< 100 messages is healthy)
- Verify `/metrics` endpoint is scrapable

### Weekly
- Run dependency audit: `make audit-deps`
- Review analyst feedback for labeling consistency
- Check model F1 score for degradation

### Monthly
- Run security scan: `make security-scan`
- Backtest champion vs challenger models
- Review CTAF compliance KPIs for overdue SARs
- Rotate API tokens

## Incident Response

1. **Assess severity**: Use Grafana alert rules
2. **Contain**: If pipeline issue, pause producer: `docker compose stop producer`
3. **Diagnose**: Check logs — `docker compose logs consumer`, `docker compose logs api`
4. **Resolve**: Apply fix, verify with test endpoint
5. **Document**: Log the incident in audit trail
