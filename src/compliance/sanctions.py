"""Sanctions and PEP screening utilities."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SanctionsScreeningResult:
    is_hit: bool
    matched_account: Optional[str] = None
    matched_field: Optional[str] = None
    list_name: Optional[str] = None
    entity_name: Optional[str] = None


class SanctionsScreener:
    """Exact-match account screening against a local sanctions CSV mirror."""

    REQUIRED_COLUMNS = {"account_id"}

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = Path(csv_path or os.getenv("SANCTIONS_CSV_PATH", "./data/sanctions/accounts.csv"))
        self._entries = self._load_entries()
        self.account_ids = frozenset(self._entries)

    def _load_entries(self) -> dict[str, dict[str, str]]:
        if not self.csv_path.exists():
            logger.warning("Sanctions CSV not found at %s; sanctions screening has no local entries", self.csv_path)
            return {}

        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])
            missing = self.REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    f"Sanctions CSV {self.csv_path} missing required columns: {', '.join(sorted(missing))}"
                )

            entries = {}
            for row in reader:
                account_id = (row.get("account_id") or "").strip()
                if not account_id:
                    continue
                entries[account_id] = {
                    "list_name": (row.get("list_name") or "LOCAL_SANCTIONS").strip(),
                    "entity_name": (row.get("entity_name") or "").strip(),
                }
            return entries

    def screen(self, tx_data: dict) -> SanctionsScreeningResult:
        for field in ("sender_account", "receiver_account", "source_account", "destination_account", "user_id"):
            account = tx_data.get(field)
            if account in self._entries:
                entry = self._entries[account]
                return SanctionsScreeningResult(
                    is_hit=True,
                    matched_account=account,
                    matched_field=field,
                    list_name=entry.get("list_name"),
                    entity_name=entry.get("entity_name"),
                )
        return SanctionsScreeningResult(is_hit=False)
