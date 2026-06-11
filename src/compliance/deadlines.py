"""Business-day deadline helpers."""

import os
from datetime import datetime, timedelta, timezone

_FIXED_TN_HOLIDAYS = {
    "01-01",
    "03-20",
    "03-21",
    "04-09",
    "05-01",
    "07-25",
    "08-13",
    "10-15",
}


def _configured_holidays() -> set[str]:
    raw = os.getenv("TUNISIA_ISLAMIC_HOLIDAYS", "")
    return {value.strip() for value in raw.split(",") if value.strip()}


def ctaf_filing_deadline(from_date: datetime | None = None, business_days: int = 10) -> datetime:
    """Count Tunisian business days after a detection date."""
    current = from_date or datetime.now(timezone.utc).replace(tzinfo=None)
    configured_holidays = _configured_holidays()
    days_counted = 0

    while days_counted < business_days:
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)
        current += timedelta(days=1)

        if current.weekday() >= 5:
            continue
        if current.strftime("%m-%d") in _FIXED_TN_HOLIDAYS:
            continue
        if current.strftime("%Y-%m-%d") in configured_holidays:
            continue

        days_counted += 1

    return current
