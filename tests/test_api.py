"""Tests for the FastAPI Command Center API."""
import pytest
import sqlite3
from datetime import datetime, timezone, timedelta


class TestHealthEndpoint:
    def test_health_check(self, api_test_client):
        response = api_test_client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAuthEndpoints:
    def test_whoami_admin(self, api_test_client, admin_headers):
        response = api_test_client.get("/auth/whoami", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "ADMIN"
        assert data["authenticated"] is True

    def test_whoami_analyst(self, api_test_client, analyst_headers):
        response = api_test_client.get("/auth/whoami", headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "ANALYST"

    def test_unauthorized_request(self, api_test_client):
        response = api_test_client.get("/auth/whoami", headers={"Authorization": "Bearer wrong_token"})
        assert response.status_code == 401


class TestAlertEndpoints:
    def test_add_alert(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_API_TEST_001",
            "user_id": "USER_5000",
            "amount_tnd": 7500.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.95,
            "alert_type": "high_risk",
            "ingestion_latency": 1.23,
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_add_alert_analyst_forbidden(self, api_test_client, analyst_headers):
        alert = {
            "transaction_id": "TXN_FORBIDDEN",
            "user_id": "USER_5000",
            "amount_tnd": 100.0,
            "governorate": "Tunis",
            "payment_method": "eDinar",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.5,
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=analyst_headers)
        assert response.status_code == 403

    def test_review_queue(self, api_test_client, admin_headers):
        # First add an alert
        alert = {
            "transaction_id": "TXN_QUEUE_001",
            "user_id": "USER_6000",
            "amount_tnd": 5000.0,
            "governorate": "Sfax",
            "payment_method": "Konnect",
            "branch_id": "Sfax-Agency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.91,
            "alert_type": "high_risk",
        }
        api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)

        # Then query the review queue
        response = api_test_client.get("/alerts/review-queue/?limit=10", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_add_alert_persists_shap_top5(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_SHAP_001",
            "user_id": "USER_SHAP",
            "amount_tnd": 9000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.96,
            "alert_type": "high_risk",
            "shap_top5": [
                {
                    "feature": "v_count",
                    "value": 5,
                    "impact": 0.41,
                    "abs_impact": 0.41,
                    "direction": "increases_risk",
                    "confidence": 0.82,
                }
            ],
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert response.status_code == 200

        explain_response = api_test_client.get("/alerts/TXN_SHAP_001/explain", headers=admin_headers)
        assert explain_response.status_code == 200
        explanation = explain_response.json()
        assert explanation["shap_top5"][0]["feature"] == "v_count"
        assert explanation["shap_top5"][0]["description"] == "High velocity (v_count)"
        assert explanation["top_risk_factors"][0]["impact"] == 0.41

    def test_add_high_anomaly_alert_persists_anomaly_fields(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_HIGH_ANOMALY_API",
            "user_id": "USER_ANOMALY",
            "amount_tnd": 1200.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.22,
            "alert_type": "HIGH_ANOMALY",
            "anomaly_score": -0.41,
            "anomaly_model_version": "iso_test",
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert response.status_code == 200

        queue = api_test_client.get(
            "/alerts/review-queue/?alert_type=HIGH_ANOMALY",
            headers=admin_headers,
        )
        assert queue.status_code == 200
        data = queue.json()
        assert any(item["transaction_id"] == "TXN_HIGH_ANOMALY_API" for item in data)


class TestFeedbackEndpoints:
    def test_submit_feedback(self, api_test_client, admin_headers, analyst_headers):
        # Add an alert first
        alert = {
            "transaction_id": "TXN_FEEDBACK_001",
            "user_id": "USER_7000",
            "amount_tnd": 3000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.93,
        }
        api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)

        # Submit feedback
        feedback = {
            "transaction_id": "TXN_FEEDBACK_001",
            "analyst_label": "Confirmed Fraud",
            "analyst_comment": "Clear fraud pattern",
        }
        response = api_test_client.post("/feedback/", json=feedback, headers=analyst_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_invalid_feedback_label(self, api_test_client, analyst_headers):
        feedback = {
            "transaction_id": "TXN_001",
            "analyst_label": "Invalid Label",
        }
        response = api_test_client.post("/feedback/", json=feedback, headers=analyst_headers)
        assert response.status_code == 422  # Validation error


class TestStatsEndpoint:
    def test_get_stats(self, api_test_client, admin_headers):
        response = api_test_client.get("/stats/", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data
        assert "high_risk_alerts" in data
        assert "precision" in data


class TestComplianceKpisEndpoint:
    def test_compliance_kpis_are_derived_from_recorded_facts(self, api_test_client, admin_headers, tmp_db):
        now = datetime.now(timezone.utc)
        overdue_detection = now - timedelta(days=45)
        recent_detection = now - timedelta(days=2)
        branch_id = "KPI-Test-Branch"

        conn = sqlite3.connect(str(tmp_db))
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO high_risk_alerts
            (transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id,
             timestamp, ml_probability, sar_report, alert_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                "TXN_KPI_SAR_ON_TIME",
                "USER_KPI_1",
                5000.0,
                "Tunis",
                "Flouci",
                branch_id,
                recent_detection.isoformat(),
                0.91,
                "SAR report text",
                "high_risk",
                now.isoformat(),
            ),
            (
                "TXN_KPI_OVERDUE",
                "USER_KPI_2",
                7000.0,
                "Sfax",
                "eDinar",
                branch_id,
                overdue_detection.isoformat(),
                0.89,
                None,
                "high_risk",
                overdue_detection.isoformat(),
            ),
            (
                "TXN_KPI_SANCTIONS",
                "USER_KPI_3",
                9000.0,
                "Sousse",
                "Konnect",
                branch_id,
                recent_detection.isoformat(),
                1.0,
                "SAR sanctions report",
                "SANCTIONS_HIT",
                now.isoformat(),
            ),
        ])
        cursor.executemany("""
            INSERT INTO feedback_labels (transaction_id, analyst_label, analyst_comment, branch_id)
            VALUES (?, ?, ?, ?)
        """, [
            ("TXN_KPI_SAR_ON_TIME", "False Positive", "Reviewed", branch_id),
            ("TXN_KPI_SANCTIONS", "Confirmed Fraud", "Reviewed", branch_id),
        ])
        cursor.execute("""
            INSERT INTO pkyc_triggers
            (event_type, account_id, trigger_reason, timestamp, current_risk_tier, signals, transaction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "pKYC_trigger",
            "hashed-account",
            "LOW_RISK_TO_HIGH_SCORE",
            now.isoformat(),
            "HIGH",
            "{}",
            "TXN_KPI_SAR_ON_TIME",
        ))
        conn.commit()
        conn.close()

        response = api_test_client.get(f"/compliance/kpis/?branch_id={branch_id}", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["sar_reports_generated"] == 2
        assert data["sar_on_time_percent"] == 100.0
        assert data["overdue_sar_count"] == 1
        assert data["overdue_sars"][0]["transaction_id"] == "TXN_KPI_OVERDUE"
        assert data["sanctions_hits"] == 1
        assert data["pkyc_triggers_by_reason"] == {"LOW_RISK_TO_HIGH_SCORE": 1}
        assert data["false_positive_rate"] == 50.0
        assert data["high_risk_accounts_by_tier"]["CRITICAL"] == 2
        assert data["branch_id"] == branch_id


class TestExplainEndpoint:
    def test_explain_nonexistent_transaction(self, api_test_client, admin_headers):
        response = api_test_client.get("/alerts/NONEXISTENT/explain", headers=admin_headers)
        assert response.status_code == 404


class TestDriftMetricsEndpoint:
    def test_drift_metrics_shape(self, api_test_client, analyst_headers):
        response = api_test_client.get("/metrics/drift", headers=analyst_headers)

        assert response.status_code == 200
        data = response.json()
        assert "psi_results" in data
        assert "score_drift" in data
        assert "decision" in data
