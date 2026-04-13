# Security Policy

## Supported Versions

| Version | Supported          |
|---------| ------------------ |
| 1.0.x   | :white_check_mark: |

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
2. **Automated Scanning**: CI runs `bandit` (Python AST security linter) and `pip-audit` (CVE database) on every PR
3. **PII Protection**: User identifiers are HMAC-SHA256 hashed before storage
4. **No Secrets in Code**: All credentials are loaded from environment variables or HashiCorp Vault
5. **Input Validation**: Pydantic schemas validate all API inputs
6. **Rate Limiting**: API endpoints are rate-limited to prevent abuse
7. **Audit Logging**: All administrative actions are logged with before/after state

### Automated Security Checks

```bash
# Run on every commit (add to CI/CD)
bandit -r src/ -f json --exit-code 2
pip-audit --requirement requirements.txt
safety check -r requirements.txt
```

### Known Limitations

- SQLite is used for the feedback database. In production, this should be replaced with PostgreSQL with TDE (Transparent Data Encryption).
- The DLQ is stored on the local filesystem. In production, this should be a persistent volume or cloud storage.
- Ollama LLM runs without authentication. In production, add network-level access control.

### Dependency Policy

- All packages must be pinned to exact versions (`package==1.2.3`)
- No `>=`, `~=`, or unpinned versions in `requirements.txt`
- Monthly review of `pip-audit` and `safety` scan results
- Critical CVEs (CVSS ≥ 9.0) must be patched within 24 hours
- High CVEs (CVSS ≥ 7.0) must be patched within 7 days
