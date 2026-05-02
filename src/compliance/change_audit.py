"""Tamper-evident audit logging for rule and model governance changes."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _default_audit_path() -> Path:
    return Path(os.getenv("CHANGE_AUDIT_LOG", "./data/audit/change_audit.jsonl"))


def _last_entry_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None

    last_hash = None
    with path.open("r", encoding="utf-8") as audit_file:
        for line in audit_file:
            if not line.strip():
                continue
            try:
                last_hash = json.loads(line).get("entry_hash")
            except json.JSONDecodeError:
                continue
    return last_hash


def append_change_audit_event(event: dict[str, Any], audit_log_path: Optional[str] = None) -> dict[str, Any]:
    """Append a hash-chained governance audit event and return the stored event."""
    path = Path(audit_log_path) if audit_log_path else _default_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    enriched = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": "change_audit.v1",
        **event,
        "previous_hash": _last_entry_hash(path),
    }
    canonical = json.dumps(enriched, sort_keys=True, default=str, separators=(",", ":"))
    enriched["entry_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    with path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(enriched, sort_keys=True, default=str) + "\n")

    return enriched
