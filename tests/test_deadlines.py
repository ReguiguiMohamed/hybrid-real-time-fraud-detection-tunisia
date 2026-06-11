"""Tests for business-day deadline calculation."""

from datetime import datetime
from unittest.mock import patch

from compliance.deadlines import ctaf_filing_deadline


def test_deadline_skips_weekends():
    start = datetime(2026, 5, 4)

    assert ctaf_filing_deadline(start, business_days=10) == datetime(2026, 5, 18)


def test_deadline_skips_fixed_holiday():
    start = datetime(2026, 4, 30)

    assert ctaf_filing_deadline(start, business_days=1) == datetime(2026, 5, 4)


def test_deadline_skips_configured_holiday():
    start = datetime(2026, 5, 4)

    with patch.dict("os.environ", {"TUNISIA_ISLAMIC_HOLIDAYS": "2026-05-05"}):
        assert ctaf_filing_deadline(start, business_days=1) == datetime(2026, 5, 6)


def test_default_deadline_is_in_the_future():
    assert ctaf_filing_deadline() > datetime.now()
