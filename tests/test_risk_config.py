"""Tests for risk configuration constants."""

from shared.risk_config import CBDC_PILOT_GOVERNORATES, D17_SOFT_LIMIT, D17_VELOCITY_CAP, RISK_WEIGHTS


class TestRiskConfig:
    def test_weights_sum_to_one(self):
        total = sum(RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Risk weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self):
        for key, val in RISK_WEIGHTS.items():
            assert val > 0, f"Weight for {key} should be positive, got {val}"

    def test_required_weight_keys(self):
        required = {"velocity", "travel", "high_value", "d17_limit"}
        assert required == set(RISK_WEIGHTS.keys())

    def test_cbdc_pilot_governorates(self):
        assert "Tunis" in CBDC_PILOT_GOVERNORATES
        assert "Sfax" in CBDC_PILOT_GOVERNORATES

    def test_d17_soft_limit(self):
        assert D17_SOFT_LIMIT == 1500.0

    def test_d17_velocity_cap(self):
        assert D17_VELOCITY_CAP == 5
