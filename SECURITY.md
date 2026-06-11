# Security Policy

## Supported Versions

| Version | Supported          |
|---------| ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, email `security@amastan.tn` with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

You will receive a response within **48 hours**. If the issue is accepted, a patch will be released within **7 days**.

## Security Posture

### What We Do

1. **Dependency Pinning**: All dependencies are pinned to exact versions in `requirements.txt`
2. **Automated Scanning**: CI runs `bandit` on every PR; scheduled workflows publish advisory `pip-audit` and Semgrep reports
3. **PII Protection**: User identifiers are HMAC-SHA256 hashed before storage
4. **No Secrets in Code**: All credentials are loaded from environment variables or HashiCorp Vault
5. **Input Validation**: Pydantic schemas validate all API inputs
6. **Rate Limiting**: API endpoints are rate-limited to prevent abuse
7. **Audit Logging**: All administrative actions are logged with before/after state

### Automated Security Checks

```bash
# Required CI gate
bandit -r src dashboard scripts -lll

# Advisory local reports
pip-audit --requirement requirements.txt
```

### Known Limitations

- Local development uses SQLite. The verified hosted prototype uses Neon PostgreSQL, but regulated deployment still requires an approved encryption, migration, backup, and access-control design.
- The DLQ is stored on the local filesystem. In production, this should be a persistent volume or cloud storage.
- Ollama LLM runs without authentication. In production, add network-level access control.

### Dependency Policy

- All packages must be pinned to exact versions (`package==1.2.3`)
- No `>=`, `~=`, or unpinned versions in `requirements.txt`
- Monthly review of `pip-audit` and `safety` scan results
- Critical CVEs (CVSS ≥ 9.0) must be patched within 24 hours
- High CVEs (CVSS ≥ 7.0) must be patched within 7 days

### Accepted Dependency Risks (with Expiry Dates)

The following risks have been reviewed, documented, and accepted for the v0.1.0 prototype:

| Dependency | Issue | Risk | Review Date | Expiry | Notes |
|---|---|---|---|---|---|
| `google.protobuf` (≥4.25.8) | Deprecated `PyType_Spec` API | Low — runtime behavior unaffected; cosmetic deprecation warning only | 2026-06-10 | 2026-09-01 | Upstream ChromaDB dependency; warning appears in ChromaDB test paths only, not in verified API slice |
| `pyspark==4.1.1` | Large dependency surface (Java JRE required at runtime) | Medium — only loaded on-demand by optional ML pipeline | 2026-06-10 | 2026-09-01 | Not installed in verified-slice CI; local/Spark workflows only |
| `chromadb==0.5.0` | Pulls unzipped models on first use; no network isolation | Medium — RAG engine is optional, not deployed on HF Space | 2026-06-10 | 2026-09-01 | Only loaded on-demand via optional import in `rag_engine/` |
| `torch==2.4.0` | Large binary (~2GB); CVE surface varies by platform | Medium — only loaded by optional `sentence-transformers` for RAG | 2026-06-10 | 2026-09-01 | Not in verified-slice CI; excluded from CI dependency set |
| `ollama==0.3.3` | Runs unsigned LLM binary; no authentication in dev mode | High on prod — not deployed on HF Space; local development only | 2026-06-10 | 2026-09-01 | `OLLAMA_HOST` must be restricted to localhost in production |
| `sentence-transformers==3.0.1` | Pulls model weights from HuggingFace at first import | Low for verified slice — never imported by API code | 2026-06-10 | 2026-09-01 | Only imported by `rag_engine/` optional module |

All accepted risks will be re-evaluated no later than the expiry date. If a fix or mitigation
becomes available sooner, the risk entry will be resolved ahead of schedule.

### Token Rotation Procedure

Before any public demonstration or production deployment, rotate all authentication tokens:

| Token | Location | Rotation Command / Procedure |
|---|---|---|
| `ADMIN_TOKEN` | HF Space Secrets → `ADMIN_TOKEN` env var | Generate new UUID; update in HF Space settings; update `.env.example` placeholder |
| `ANALYST_TOKEN` | HF Space Secrets → `ANALYST_TOKEN` env var | Generate new UUID; update in HF Space settings; update `.env.example` placeholder |
| `METRICS_TOKEN` | HF Space Secrets + Grafana Cloud | Generate new token; update both HF Space and Grafana Cloud integration simultaneously |
| `HF_TOKEN` | GitHub Secrets → `HF_TOKEN` | Generate HF User Access Token; update GitHub repo secret |
| `PII_SALT_KEY` | HF Space Secrets → `PII_SALT_KEY` env var | Generate 32-char random hex string; existing hashes will mismatch after rotation (plan for migration window) |

**Minimum rotation cadence**: every 90 days for production deployments.

**Rotation steps for HF Space**:
1. Generate new token value (use `python -c "import uuid; print(uuid.uuid4().hex)"`)
2. Update the environment variable in Hugging Face Space Settings → Repository Secrets
3. If `METRICS_TOKEN`, update Grafana Cloud integration simultaneously
4. Verify: `curl -I -H "Authorization: Bearer <new-token>" https://mohamedreg-amastan-fraud-shield-api.hf.space/api/v1/auth/whoami`
5. Revoke the old token after 24-hour cooldown period
