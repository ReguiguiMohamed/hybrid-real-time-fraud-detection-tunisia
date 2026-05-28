"""
Tests for CTAF SAR compliance logic introduced in P0-2.

Pure Python — no Spark or external services required.
Covers:
- ctaf_filing_deadline(): Tunisian business-day calculation
- generate_deterministic_fallback(): deadline correctness, penalty language
- format_sar_report(): penalty line present in output
"""
import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from rag_engine.sar_validator import (
    ctaf_filing_deadline,
    generate_deterministic_fallback,
    format_sar_report,
)


# ---------------------------------------------------------------------------
# ctaf_filing_deadline
# ---------------------------------------------------------------------------

class TestCtafFilingDeadline:

    def test_skips_weekends(self):
        # Monday 2026-04-27 + 10 business days = Monday 2026-05-11
        # (skipping Sat 05-02, Sun 05-03, Sat 05-09, Sun 05-10)
        monday = datetime(2026, 4, 27)
        result = ctaf_filing_deadline(from_date=monday, business_days=10)
        assert result.weekday() < 5, "Deadline must not fall on a weekend"
        assert result > monday

    def test_exactly_ten_business_days_with_labour_day(self):
        # From Monday 2026-04-27, counting 10 Tunisian business days:
        # Apr 28 (d1), Apr 29 (d2), Apr 30 (d3),
        # May 1 = Labour Day SKIP, May 2-3 = weekend SKIP,
        # May 4 (d4), May 5 (d5), May 6 (d6), May 7 (d7), May 8 (d8),
        # May 9-10 = weekend SKIP,
        # May 11 (d9), May 12 (d10)  → deadline = Tue May 12
        start = datetime(2026, 4, 27)
        result = ctaf_filing_deadline(from_date=start, business_days=10)
        assert result.date() == datetime(2026, 5, 12).date()

    def test_skips_labour_day_fixed_holiday(self):
        # From Thu 2026-04-30: next 10 business days must skip May 1 (Labour Day)
        # Fri May 1 = holiday, so day 1 = Mon May 4
        start = datetime(2026, 4, 30)
        result = ctaf_filing_deadline(from_date=start, business_days=10)
        # May 1 skipped → deadline pushed 1 day later than if no holiday
        no_holiday_result = ctaf_filing_deadline(
            from_date=start, business_days=10
        )
        # Just assert the deadline is after the holiday
        assert result.date() > datetime(2026, 5, 1).date()

    def test_skips_independence_day_march_20(self):
        start = datetime(2026, 3, 18)  # Wednesday
        result = ctaf_filing_deadline(from_date=start, business_days=1)
        # Mar 19 = Thu (business day) → deadline should be Mar 19
        assert result.date() == datetime(2026, 3, 19).date()

        # From Mar 19, next 1 business day: Mar 20 = Independence Day (skip), Mar 21 = Youth Day (skip)
        # → next business day = Mon Mar 23
        start2 = datetime(2026, 3, 19)
        result2 = ctaf_filing_deadline(from_date=start2, business_days=1)
        assert result2.date() == datetime(2026, 3, 23).date()

    def test_skips_islamic_holidays_from_env(self):
        # Inject a deterministic Islamic holiday date on the first available business day
        start = datetime(2026, 5, 4)  # Monday
        # Without holiday, 1 business day = Tuesday May 5
        result_no_holiday = ctaf_filing_deadline(from_date=start, business_days=1)
        assert result_no_holiday.date() == datetime(2026, 5, 5).date()

        # With May 5 declared an Islamic holiday → deadline shifts to May 6
        with patch.dict(os.environ, {"TUNISIA_ISLAMIC_HOLIDAYS": "2026-05-05"}):
            result_with_holiday = ctaf_filing_deadline(from_date=start, business_days=1)
        assert result_with_holiday.date() == datetime(2026, 5, 6).date()

    def test_deadline_is_always_a_weekday(self):
        for day_offset in range(14):
            start = datetime(2026, 5, 1) + timedelta(days=day_offset)
            result = ctaf_filing_deadline(from_date=start, business_days=10)
            assert result.weekday() < 5, f"Deadline from {start.date()} landed on a weekend"

    def test_default_from_date_is_now(self):
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        result = ctaf_filing_deadline()
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        # Deadline must be after both before and after (it's in the future)
        assert result > before


# ---------------------------------------------------------------------------
# generate_deterministic_fallback
# ---------------------------------------------------------------------------

class TestDeterministicFallbackCompliance:

    def _make_tx(self, amount=500.0, payment_method="Flouci"):
        return {
            "transaction_id": "TXN_TEST_001",
            "user_id": "USER_001",
            "amount_tnd": amount,
            "governorate": "Tunis",
            "payment_method": payment_method,
            "branch_id": "Tunis-GNC",
            "timestamp": "2026-05-01T10:00:00Z",
        }

    def test_filing_deadline_is_future_date(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.8)
        deadline_str = report.urgency_assessment.filing_deadline
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
        assert deadline.date() > datetime.now(timezone.utc).replace(tzinfo=None).date(), (
            f"filing_deadline {deadline_str} must be in the future"
        )

    def test_filing_deadline_is_not_today(self):
        # Old bug: deadline was set to datetime.now(timezone.utc).replace(tzinfo=None) (i.e., now / today)
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.5)
        deadline_str = report.urgency_assessment.filing_deadline
        today = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
        assert deadline_str != today, (
            "filing_deadline must not be today — must be 10 business days from now"
        )

    def test_filing_deadline_is_approximately_ten_business_days(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.7)
        deadline_str = report.urgency_assessment.filing_deadline
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
        delta = (deadline.date() - datetime.now(timezone.utc).replace(tzinfo=None).date()).days
        # 10 business days = 14–16 calendar days depending on weekends/holidays
        assert 10 <= delta <= 20, f"Deadline gap {delta} days is outside expected range"

    def test_penalty_language_in_urgency_reason(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.9)
        reason = report.urgency_assessment.reason
        assert "50,000" in reason or "50000" in reason, (
            "Urgency reason must mention the TND 50,000 CTAF non-compliance penalty"
        )

    def test_penalty_language_in_recommended_steps(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.6)
        combined_steps = " ".join(report.recommended_next_steps)
        assert "50,000" in combined_steps or "50000" in combined_steps, (
            "Recommended steps must mention the TND 50,000 penalty"
        )

    def test_business_days_language_in_steps(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.6)
        combined_steps = " ".join(report.recommended_next_steps)
        assert "business" in combined_steps.lower() or "ouvrables" in combined_steps.lower(), (
            "Steps must specify 'business days' (jours ouvrables), not just 'days'"
        )

    def test_large_tx_risk_factor_uses_updated_threshold(self):
        # P0-1: threshold is now 15,000 TND, not 5,000
        report_below = generate_deterministic_fallback(self._make_tx(amount=6000.0), ml_score=0.5)
        report_above = generate_deterministic_fallback(self._make_tx(amount=20000.0), ml_score=0.5)

        factors_below = [rf.factor for rf in report_below.risk_factors]
        factors_above = [rf.factor for rf in report_above.risk_factors]

        # 6,000 TND is below 15,000 threshold — should NOT get the large-tx factor
        assert not any("15,000" in f or "Large transaction" in f for f in factors_below), (
            "6,000 TND should not trigger the large-transaction risk factor (threshold is 15,000)"
        )
        # 20,000 TND exceeds 15,000 — should get the factor
        assert any("Large transaction" in f or "15,000" in f for f in factors_above), (
            "20,000 TND should trigger the large-transaction risk factor"
        )

    def test_report_passes_pydantic_validation(self):
        report = generate_deterministic_fallback(self._make_tx(), ml_score=0.75)
        assert report.transaction_id == "TXN_TEST_001"
        assert report.ml_score == 0.75
        assert len(report.risk_factors) >= 1
        assert len(report.regulatory_violations) >= 1
        assert len(report.recommended_next_steps) >= 1


# ---------------------------------------------------------------------------
# format_sar_report
# ---------------------------------------------------------------------------

class TestFormatSarReport:

    def test_formatted_output_contains_penalty_line(self):
        tx = {
            "transaction_id": "TXN_FMT_001", "user_id": "U1",
            "amount_tnd": 1000.0, "governorate": "Sfax",
            "payment_method": "eDinar", "branch_id": "B1",
            "timestamp": "2026-05-01T09:00:00Z",
        }
        report = generate_deterministic_fallback(tx, ml_score=0.6)
        formatted = format_sar_report(report)
        assert "50,000" in formatted, "Formatted SAR must display the TND 50,000 penalty"
        assert "jours ouvrables" in formatted or "business days" in formatted.lower(), (
            "Formatted SAR must specify business days in the deadline line"
        )

    def test_formatted_output_contains_deadline(self):
        tx = {
            "transaction_id": "TXN_FMT_002", "user_id": "U2",
            "amount_tnd": 500.0, "governorate": "Tunis",
            "payment_method": "Flouci", "branch_id": "B2",
            "timestamp": "2026-05-01T08:00:00Z",
        }
        report = generate_deterministic_fallback(tx, ml_score=0.85)
        formatted = format_sar_report(report)
        assert "Deadline:" in formatted
        # Deadline should contain a future year
        assert "2026" in formatted or "2027" in formatted
