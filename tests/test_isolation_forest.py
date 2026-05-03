import pandas as pd

from ml.isolation_forest import IsolationForestAnomalyDetector


def training_frame():
    return pd.DataFrame([
        {"amount_tnd": 100.0, "ml_probability": 0.05, "risk_score": 0.05, "v_count": 1, "g_dist": 1},
        {"amount_tnd": 120.0, "ml_probability": 0.06, "risk_score": 0.06, "v_count": 1, "g_dist": 1},
        {"amount_tnd": 90.0, "ml_probability": 0.04, "risk_score": 0.04, "v_count": 1, "g_dist": 1},
        {"amount_tnd": 130.0, "ml_probability": 0.07, "risk_score": 0.07, "v_count": 2, "g_dist": 1},
        {"amount_tnd": 110.0, "ml_probability": 0.05, "risk_score": 0.05, "v_count": 1, "g_dist": 1},
        {"amount_tnd": 10000.0, "ml_probability": 0.2, "risk_score": 0.2, "v_count": 12, "g_dist": 4},
    ])


def test_isolation_forest_scores_records_and_flags_high_anomaly():
    detector = IsolationForestAnomalyDetector(contamination=0.2).train(training_frame(), "iso_test")

    result = detector.score([{
        "amount_tnd": 15000.0,
        "ml_probability": 0.2,
        "risk_score": 0.2,
        "v_count": 20,
        "g_dist": 5,
    }])[0]

    assert result["model_version"] == "iso_test"
    assert "anomaly_score" in result
    assert result["is_anomaly"] is True


def test_isolation_forest_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "isolation_forest.joblib"
    detector = IsolationForestAnomalyDetector(contamination=0.2).train(training_frame(), "iso_roundtrip")
    detector.save(path)

    loaded = IsolationForestAnomalyDetector.load(path)
    result = loaded.score([{
        "amount_tnd": 100.0,
        "ml_probability": 0.05,
        "risk_score": 0.05,
        "v_count": 1,
        "g_dist": 1,
    }])[0]

    assert loaded.model_version == "iso_roundtrip"
    assert result["model_version"] == "iso_roundtrip"


def test_metadata_json_contains_threshold():
    detector = IsolationForestAnomalyDetector(contamination=0.2).train(training_frame(), "iso_meta")

    metadata = detector.metadata_json()

    assert "high_anomaly_threshold" in metadata
    assert "iso_meta" in metadata
