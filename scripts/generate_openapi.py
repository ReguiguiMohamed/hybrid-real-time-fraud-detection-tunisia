#!/usr/bin/env python3
"""Generate openapi.json from the FastAPI app and write it to docs/."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("ANALYST_TOKEN", "openapi-generation-only")
os.environ.setdefault("ADMIN_TOKEN", "openapi-generation-only-admin")

from dashboard.api import app

output_path = ROOT / "docs" / "openapi.json"
spec = app.openapi()
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=2)
print(f"OpenAPI spec written to {output_path}")
print(f"Paths: {len(spec.get('paths', {}))}")
