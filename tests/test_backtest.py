"""Focused tests for deterministic backtest behavior."""

import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from scripts.backtest import BacktestEngine


def test_backtest_artifact_script_runs_from_any_working_directory(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate_backtest_artifact.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "backtest-report.json").is_file()


@pytest.fixture
def labeled_history():
    return pd.DataFrame(
        [
            {
                "amount_tnd": 16000.0,
                "payment_method": "card",
                "v_count": 1,
                "g_dist": 1,
                "label": 1,
            },
            {
                "amount_tnd": 100.0,
                "payment_method": "card",
                "v_count": 1,
                "g_dist": 1,
                "label": 0,
            },
        ]
    )


def test_alert_threshold_is_applied(labeled_history):
    engine = BacktestEngine()

    low_threshold = engine._apply_original_rules(labeled_history, alert_threshold=0.1)
    high_threshold = engine._apply_original_rules(labeled_history, alert_threshold=0.9)

    assert low_threshold["original_alert"].sum() == 1
    assert high_threshold["original_alert"].sum() == 0


def test_modified_threshold_uses_shared_scoring(labeled_history):
    engine = BacktestEngine()

    scored = engine._apply_modified_rules(
        labeled_history,
        threshold_changes={"high_value_threshold": 20000.0},
    )

    assert scored["high_value_risk"].sum() == 0
    assert scored["modified_alert"].sum() == 0


def test_proxy_labels_never_produce_deployment_recommendation(monkeypatch):
    history = pd.DataFrame(
        [
            {
                "amount_tnd": 16000.0,
                "payment_method": "card",
                "v_count": 1,
                "g_dist": 1,
                "ml_probability": 0.95,
            }
        ]
    )
    engine = BacktestEngine()
    monkeypatch.setattr(engine, "load_data", lambda *_args, **_kwargs: history)

    result = engine.run()

    assert result.label_source == "ml_probability_proxy"
    assert result.recommendation.startswith("NON-DECISIONAL")


def test_missing_labels_raise_clear_error():
    engine = BacktestEngine()
    history = pd.DataFrame([{"amount_tnd": 100.0, "payment_method": "card", "v_count": 1, "g_dist": 1}])
    scored = engine._apply_original_rules(history)

    with pytest.raises(ValueError, match="verified label"):
        engine._compute_metrics(scored, "original_score", "original_alert")


def test_empty_input_raises_clear_error():
    engine = BacktestEngine()
    empty = pd.DataFrame()
    with pytest.raises((ValueError, KeyError)):
        engine._apply_original_rules(empty)


def test_empty_data_raises_value_error(monkeypatch):
    engine = BacktestEngine()
    monkeypatch.setattr(engine, "load_data", lambda *_args, **_kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="No data available"):
        engine.run()


def test_model_artifact_fallback_does_not_crash(monkeypatch, tmp_path):
    """Champion model artifact fallback — when no model path is available,
    the backtest should still complete with rule-based scoring only."""
    engine = BacktestEngine()
    history = pd.DataFrame(
        [
            {"amount_tnd": 500.0, "payment_method": "card", "v_count": 1, "g_dist": 1, "label": 0},
            {"amount_tnd": 20000.0, "payment_method": "card", "v_count": 1, "g_dist": 1, "label": 1},
        ]
    )
    monkeypatch.setattr(engine, "load_data", lambda *_args, **_kwargs: history)
    result = engine.run()
    assert result.total_transactions == 2
    assert result.label_source is not None
