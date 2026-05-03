"""Pure-Python quality-gate semantics that run without Spark."""

from shared.quality_gates import evaluate_device_behavior_flags


def test_vpn_new_device_high_amount_flag():
    flags = evaluate_device_behavior_flags(
        amount_tnd=1500.0,
        device_id="DEV1",
        vpn_detected=True,
        emulator_detected=False,
        device_age_days=0.0,
        device_account_count_7d=1.0,
    )

    assert flags == {
        "device_vpn_new_high_amount_flag": True,
        "device_emulator_flag": False,
        "device_shared_accounts_flag": False,
    }


def test_emulator_flag():
    flags = evaluate_device_behavior_flags(
        amount_tnd=200.0,
        device_id="DEV2",
        vpn_detected=False,
        emulator_detected=True,
        device_age_days=30.0,
        device_account_count_7d=1.0,
    )

    assert flags["device_emulator_flag"] is True


def test_shared_device_account_velocity_flag():
    flags = evaluate_device_behavior_flags(
        amount_tnd=200.0,
        device_id="DEV3",
        vpn_detected=False,
        emulator_detected=False,
        device_age_days=10.0,
        device_account_count_7d=4.0,
    )

    assert flags["device_shared_accounts_flag"] is True


def test_old_non_vpn_device_no_flag():
    flags = evaluate_device_behavior_flags(
        amount_tnd=1500.0,
        device_id="DEV4",
        vpn_detected=False,
        emulator_detected=False,
        device_age_days=30.0,
        device_account_count_7d=1.0,
    )

    assert flags == {
        "device_vpn_new_high_amount_flag": False,
        "device_emulator_flag": False,
        "device_shared_accounts_flag": False,
    }
