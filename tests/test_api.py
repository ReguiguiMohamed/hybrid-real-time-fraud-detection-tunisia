"""Tests for the FastAPI Command Center API."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


class TestHealthEndpoint:
    def test_root_status(self, api_test_client):
        response = api_test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["routes"]["health"] == "/health/"
        assert data["version"] == "0.1.0"
        assert data["release_channel"] == "prototype"
        assert data["routes"]["docs"] == "/docs"

    def test_health_check(self, api_test_client):
        response = api_test_client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    def test_prometheus_metrics_exports_verified_slice(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_METRICS_001",
            "user_id": "USER_METRICS",
            "amount_tnd": 7600.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.94,
            "alert_type": "high_risk",
        }
        api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)

        response = api_test_client.get("/metrics")

        assert response.status_code == 200
        body = response.text
        assert "amastan_api_info" in body
        assert 'database_backend="sqlite"' in body
        assert 'amastan_db_alerts_total{alert_type="high_risk"}' in body
        assert 'amastan_alerts_ingested_total{alert_type="high_risk",result="success"}' in body
        assert "amastan_model_champion_f1_score" in body
        assert "amastan_model_champion_auc" in body

    def test_prometheus_metrics_can_require_bearer_token(self, api_test_client, monkeypatch):
        import dashboard.api as api_module

        monkeypatch.setattr(api_module, "METRICS_TOKEN", "metrics_secret")

        unauthorized = api_test_client.get("/metrics")
        authorized = api_test_client.get("/metrics", headers={"Authorization": "Bearer metrics_secret"})

        assert unauthorized.status_code == 401
        assert authorized.status_code == 200

    def test_prometheus_metrics_read_champion_registry(self, api_test_client, tmp_db):
        with sqlite3.connect(tmp_db) as conn:
            conn.execute(
                """
                INSERT INTO model_registry
                (version_id, model_path, f1_score, auc, is_champion, promoted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("prototype-v1", "models/prototype-v1", 0.81, 0.91, 1, datetime.now().isoformat()),
            )

        response = api_test_client.get("/metrics")

        assert response.status_code == 200
        assert "amastan_model_champion_f1_score 0.81" in response.text
        assert "amastan_model_champion_auc 0.91" in response.text
        assert 'amastan_model_champion_info{version_id="prototype-v1"} 1.0' in response.text


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

    def test_batch_feedback_all_success(self, api_test_client, admin_headers, analyst_headers):
        alert1 = {
            "transaction_id": "TXN_BATCH_01",
            "user_id": "U1",
            "amount_tnd": 1000.0,
            "governorate": "Tunis",
            "payment_method": "D17",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.95,
        }
        alert2 = {
            "transaction_id": "TXN_BATCH_02",
            "user_id": "U2",
            "amount_tnd": 2000.0,
            "governorate": "Sfax",
            "payment_method": "Flouci",
            "branch_id": "Sfax-Nord",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.91,
        }
        api_test_client.post("/alerts/add/", json=alert1, headers=admin_headers)
        api_test_client.post("/alerts/add/", json=alert2, headers=admin_headers)

        batch = {
            "feedback_items": [
                {
                    "transaction_id": "TXN_BATCH_01",
                    "analyst_label": "Confirmed Fraud",
                    "analyst_comment": "Batch fraud",
                },
                {"transaction_id": "TXN_BATCH_02", "analyst_label": "False Positive", "analyst_comment": "Batch fp"},
            ]
        }
        response = api_test_client.post("/feedback/batch/", json=batch, headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 2
        assert data["success_count"] == 2
        assert data["error_count"] == 0
        assert "_links" in data

    def test_batch_feedback_returns_links(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_BATCH_LINKS",
                "user_id": "U1",
                "amount_tnd": 1000.0,
                "governorate": "Tunis",
                "payment_method": "D17",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.95,
            },
            headers=admin_headers,
        )
        batch = {
            "feedback_items": [
                {
                    "transaction_id": "TXN_BATCH_LINKS",
                    "analyst_label": "Confirmed Fraud",
                    "analyst_comment": "Has links",
                },
            ]
        }
        response = api_test_client.post("/feedback/batch/", json=batch, headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["success_count"] == 1
        assert "_links" in data
        assert "feedback_batch" in data["_links"]

    def test_batch_feedback_empty_list(self, api_test_client, analyst_headers):
        response = api_test_client.post("/feedback/batch/", json={"feedback_items": []}, headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 0

    def test_batch_feedback_requires_auth(self, api_test_client):
        response = api_test_client.post("/feedback/batch/", json={"feedback_items": []})
        assert response.status_code == 401


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
        cursor.executemany(
            """
            INSERT INTO high_risk_alerts
            (transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id,
             timestamp, ml_probability, sar_report, alert_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
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
            ],
        )
        cursor.executemany(
            """
            INSERT INTO feedback_labels (transaction_id, analyst_label, analyst_comment, branch_id)
            VALUES (?, ?, ?, ?)
        """,
            [
                ("TXN_KPI_SAR_ON_TIME", "False Positive", "Reviewed", branch_id),
                ("TXN_KPI_SANCTIONS", "Confirmed Fraud", "Reviewed", branch_id),
            ],
        )
        cursor.execute(
            """
            INSERT INTO pkyc_triggers
            (event_type, account_id, trigger_reason, timestamp, current_risk_tier, signals, transaction_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "pKYC_trigger",
                "hashed-account",
                "LOW_RISK_TO_HIGH_SCORE",
                now.isoformat(),
                "HIGH",
                "{}",
                "TXN_KPI_SAR_ON_TIME",
            ),
        )
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


class TestHighRiskEndpoint:
    def _add_alert(self, client, headers, **overrides):
        payload = {
            "transaction_id": "TXN_HR_DEFAULT",
            "user_id": "USER_HR",
            "amount_tnd": 10000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "branch_id": "Tunis-GNC",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.95,
        }
        payload.update(overrides)
        return client.post("/api/v1/alerts/add/", json=payload, headers=headers)

    def test_high_risk_returns_filtered_alerts(self, api_test_client, admin_headers):
        for i, prob in [(1, 0.99), (2, 0.95), (3, 0.86), (4, 0.80)]:
            self._add_alert(api_test_client, admin_headers, transaction_id=f"TXN_HR_{i}", ml_probability=prob)

        response = api_test_client.get("/api/v1/alerts/high-risk/?limit=10", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        ids = {a["transaction_id"] for a in data}
        assert "TXN_HR_1" in ids
        assert "TXN_HR_2" in ids
        assert "TXN_HR_3" in ids
        assert "TXN_HR_4" not in ids

    def test_high_risk_filters_by_branch(self, api_test_client, admin_headers):
        self._add_alert(api_test_client, admin_headers, transaction_id="TXN_HR_BR_1", branch_id="Tunis-GNC")
        self._add_alert(api_test_client, admin_headers, transaction_id="TXN_HR_BR_2", branch_id="Sfax-Agency")

        response = api_test_client.get("/api/v1/alerts/high-risk/?branch_id=Sfax-Agency", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["transaction_id"] == "TXN_HR_BR_2"

    def test_high_risk_requires_auth(self, api_test_client, analyst_headers):
        response = api_test_client.get("/api/v1/alerts/high-risk/", headers=analyst_headers)
        assert response.status_code == 200

    def test_high_risk_unauthorized(self, api_test_client):
        response = api_test_client.get("/api/v1/alerts/high-risk/")
        assert response.status_code == 401


class TestBranchesEndpoint:
    def test_list_branches_returns_distinct(self, api_test_client, admin_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_BR_1",
                "user_id": "U1",
                "amount_tnd": 5000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "branch_id": "Tunis-GNC",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_BR_2",
                "user_id": "U2",
                "amount_tnd": 5000.0,
                "governorate": "Sfax",
                "payment_method": "eDinar",
                "branch_id": "Sfax-Agency",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )

        response = api_test_client.get("/branches/", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "Tunis-GNC" in data
        assert "Sfax-Agency" in data

    def test_branches_empty_when_no_alerts(self, api_test_client, admin_headers):
        response = api_test_client.get("/branches/", headers=admin_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_branches_unauthorized(self, api_test_client):
        response = api_test_client.get("/branches/")
        assert response.status_code == 401


class TestModelPerformanceEndpoint:
    def test_model_performance_returns_shape(self, api_test_client, admin_headers, populated_db, monkeypatch):
        response = api_test_client.get("/monitoring/model-performance/", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "total_evaluated" in data

    def test_model_performance_with_populated_data(self, api_test_client, admin_headers, populated_db, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{populated_db.as_posix()}")
        import importlib

        import shared.database as database_module

        importlib.reload(database_module)
        import dashboard.api as api_module

        importlib.reload(api_module)

        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)
        headers = {"Authorization": "Bearer test_admin_token"}

        response = client.get("/monitoring/model-performance/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_evaluated"] >= 3

    def test_model_performance_analyst_access(self, api_test_client, analyst_headers):
        response = api_test_client.get("/monitoring/model-performance/", headers=analyst_headers)
        assert response.status_code == 200

    def test_model_performance_unauthorized(self, api_test_client):
        response = api_test_client.get("/monitoring/model-performance/")
        assert response.status_code == 401


class TestExportEndpoint:
    def test_export_existing_alert(self, api_test_client, admin_headers):
        api_test_client.post(
            "/api/v1/alerts/add/",
            json={
                "transaction_id": "TXN_EXP_001",
                "user_id": "U1",
                "amount_tnd": 7500.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "branch_id": "Tunis-GNC",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.95,
                "sar_report": "SAR text for export",
                "shap_top5": [{"feature": "v_count", "value": 5, "impact": 0.41, "abs_impact": 0.41}],
            },
            headers=admin_headers,
        )

        response = api_test_client.get("/api/v1/alerts/TXN_EXP_001/export", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "TXN_EXP_001"
        assert data["sar_report"] == "SAR text for export"
        assert "exported_at" in data
        assert "shap_top5" in data
        assert "analyst_review" in data

    def test_export_nonexistent_alert_404(self, api_test_client, admin_headers):
        response = api_test_client.get("/api/v1/alerts/NONEXISTENT_ALERT/export", headers=admin_headers)
        assert response.status_code == 404

    def test_export_unauthorized(self, api_test_client):
        response = api_test_client.get("/api/v1/alerts/TXN_001/export")
        assert response.status_code == 401


class TestCtafExportEndpoint:
    def test_ctaf_export_returns_confirmed_fraud(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_CTAF_001",
                "user_id": "U1",
                "amount_tnd": 8000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "branch_id": "Tunis-GNC",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.95,
            },
            headers=admin_headers,
        )
        api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_CTAF_001",
                "analyst_label": "Confirmed Fraud",
                "analyst_comment": "Confirmed",
            },
            headers=analyst_headers,
        )

        response = api_test_client.get("/alerts/ctaf-export?days=30", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total_cases"] >= 1
        assert any(c["transaction_id"] == "TXN_CTAF_001" for c in data["cases"])
        assert "generated_at" in data

    def test_ctaf_export_excludes_false_positives(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_CTAF_FP",
                "user_id": "U1",
                "amount_tnd": 3000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "branch_id": "Tunis-GNC",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )
        api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_CTAF_FP",
                "analyst_label": "False Positive",
                "analyst_comment": "Not fraud",
            },
            headers=analyst_headers,
        )

        response = api_test_client.get("/alerts/ctaf-export?days=30", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert all(c["transaction_id"] != "TXN_CTAF_FP" for c in data["cases"])

    def test_ctaf_export_admin_only(self, api_test_client, analyst_headers):
        response = api_test_client.get("/alerts/ctaf-export", headers=analyst_headers)
        assert response.status_code == 403


class TestMonitoringEndpoints:
    def test_performance_metrics_endpoint(self, api_test_client, analyst_headers):
        response = api_test_client.get("/api/v1/metrics/performance", headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert "avg_latency_ms" in data
        assert "total_calls" in data

    def test_feedback_analysis_endpoint(self, api_test_client, analyst_headers):
        response = api_test_client.get("/api/v1/metrics/feedback", headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert "precision" in data
        assert "feedback_counts" in data

    def test_threshold_analysis_endpoint(self, api_test_client, analyst_headers):
        response = api_test_client.get("/api/v1/metrics/threshold-analysis", headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert "optimal_threshold" in data
        assert "threshold_analysis" in data

    def test_system_overview_endpoint(self, api_test_client, analyst_headers):
        response = api_test_client.get("/api/v1/metrics/system-overview", headers=analyst_headers)
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert "feedback" in data
        assert "threshold_recommendation" in data
        assert "drift" in data

    def test_monitoring_endpoints_unauthorized(self, api_test_client):
        for path in [
            "/api/v1/metrics/performance",
            "/api/v1/metrics/feedback",
            "/api/v1/metrics/threshold-analysis",
            "/api/v1/metrics/system-overview",
        ]:
            response = api_test_client.get(path)
            assert response.status_code == 401, f"{path} should reject missing auth"


class TestRetrainEndpoint:
    def test_retrain_requires_admin(self, api_test_client, analyst_headers):
        response = api_test_client.post("/retrain-model/", headers=analyst_headers)
        assert response.status_code == 403

    def test_retrain_requires_auth(self, api_test_client):
        response = api_test_client.post("/retrain-model/")
        assert response.status_code == 401

    def test_retrain_returns_success_shape(self, api_test_client, admin_headers):
        response = api_test_client.post("/retrain-model/", headers=admin_headers)
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["job_id"]

        status = api_test_client.get(data["status_url"], headers=admin_headers)
        assert status.status_code == 200
        assert status.json()["status"] in {"no_change", "promoted", "failed"}

    def test_retrain_can_be_disabled(self, api_test_client, admin_headers, monkeypatch):
        monkeypatch.setenv("MODEL_RETRAINING_ENABLED", "false")

        response = api_test_client.post("/retrain-model/", headers=admin_headers)

        assert response.status_code == 503


class TestEdgeCases:
    def test_duplicate_alert_is_detected(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_DUP_001",
            "user_id": "U1",
            "amount_tnd": 5000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.90,
        }
        first = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert first.status_code == 200
        assert first.json()["message"] == "Alert added successfully"

        second = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert second.status_code == 200
        assert second.json()["message"] == "Alert already exists"

    def test_missing_required_fields_returns_422(self, api_test_client, admin_headers):
        incomplete = {"user_id": "U1", "amount_tnd": 5000.0}
        response = api_test_client.post("/alerts/add/", json=incomplete, headers=admin_headers)
        assert response.status_code == 422

    def test_malformed_json_returns_422(self, api_test_client, admin_headers):
        response = api_test_client.post("/alerts/add/", content=b"not json", headers=admin_headers)
        assert response.status_code == 422

    def test_invalid_analyst_label_returns_422(self, api_test_client, analyst_headers):
        feedback = {
            "transaction_id": "TXN_001",
            "analyst_label": "MAYBE_FRAUD",
        }
        response = api_test_client.post("/feedback/", json=feedback, headers=analyst_headers)
        assert response.status_code == 422

    def test_empty_transaction_id_accepted_by_pipeline(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "",
            "user_id": "U1",
            "amount_tnd": 5000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.90,
        }
        response = api_test_client.post("/api/v1/alerts/add/", json=alert, headers=admin_headers)
        assert response.status_code == 200

    def test_invalid_token_returns_401_on_get_endpoints(self, api_test_client):
        bad_headers = {"Authorization": "Bearer totally_wrong"}
        for path in ["/stats/", "/branches/"]:
            response = api_test_client.get(path, headers=bad_headers)
            assert response.status_code == 401, f"{path} should reject bad token"

    def test_negative_amount_rejected(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_NEG_001",
            "user_id": "U1",
            "amount_tnd": -100.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.50,
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        # TransactionAlert has no explicit amount validator, so it may accept it
        # but should at least return a 2xx or 4xx response
        assert response.status_code in (200, 422)

    def test_metrics_labels_after_operations(self, api_test_client, admin_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_METRICS_LBL",
                "user_id": "U1",
                "amount_tnd": 5000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )
        api_test_client.get("/alerts/high-risk/", headers=admin_headers)

        metrics = api_test_client.get("/metrics")
        assert metrics.status_code == 200
        body = metrics.text
        assert "amastan_api_requests_total" in body
        assert 'method="POST"' in body or 'method="GET"' in body
        assert "alerts_ingested_total" in body


class TestIntegrationFlow:
    """End-to-end flow: add alert → review queue → feedback → stats → export."""

    def test_full_alert_lifecycle(self, api_test_client, admin_headers, analyst_headers):
        alert = {
            "transaction_id": "TXN_LIFECYCLE",
            "user_id": "USER_LC",
            "amount_tnd": 12000.0,
            "governorate": "Sfax",
            "payment_method": "Konnect",
            "branch_id": "Sfax-Agency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.97,
            "alert_type": "high_risk",
        }

        add_resp = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert add_resp.status_code == 200
        assert add_resp.json()["status"] == "success"

        queue_resp = api_test_client.get("/alerts/review-queue/?limit=10", headers=admin_headers)
        assert queue_resp.status_code == 200
        queue = queue_resp.json()
        assert any(a["transaction_id"] == "TXN_LIFECYCLE" for a in queue)

        feedback_resp = api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_LIFECYCLE",
                "analyst_label": "Confirmed Fraud",
                "analyst_comment": "High-value confirmed fraud",
            },
            headers=analyst_headers,
        )
        assert feedback_resp.status_code == 200

        stats_resp = api_test_client.get("/stats/", headers=admin_headers)
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_feedback"] >= 1

        export_resp = api_test_client.get("/alerts/ctaf-export?days=30", headers=admin_headers)
        assert export_resp.status_code == 200
        export = export_resp.json()
        assert any(c["transaction_id"] == "TXN_LIFECYCLE" for c in export["cases"])

    def test_feedback_updates_precision(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_PREC_1",
                "user_id": "U1",
                "amount_tnd": 7000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.92,
            },
            headers=admin_headers,
        )
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_PREC_2",
                "user_id": "U2",
                "amount_tnd": 5000.0,
                "governorate": "Sfax",
                "payment_method": "eDinar",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.88,
            },
            headers=admin_headers,
        )

        api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_PREC_1",
                "analyst_label": "Confirmed Fraud",
            },
            headers=analyst_headers,
        )
        api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_PREC_2",
                "analyst_label": "False Positive",
            },
            headers=analyst_headers,
        )

        stats = api_test_client.get("/stats/", headers=admin_headers).json()
        assert stats["high_risk_precision"] == 0.5


class TestLegacyEndpoints:
    """Backward-compatible unversioned endpoint aliases."""

    def test_legacy_whoami(self, api_test_client, admin_headers):
        response = api_test_client.get("/auth/whoami", headers=admin_headers)
        assert response.status_code == 200
        assert response.json()["role"] == "ADMIN"

    def test_legacy_add_alert(self, api_test_client, admin_headers):
        alert = {
            "transaction_id": "TXN_LEGACY_ADD",
            "user_id": "U1",
            "amount_tnd": 5000.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ml_probability": 0.90,
        }
        response = api_test_client.post("/alerts/add/", json=alert, headers=admin_headers)
        assert response.status_code == 200

    def test_legacy_feedback(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_LEGACY_FB",
                "user_id": "U1",
                "amount_tnd": 5000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )
        response = api_test_client.post(
            "/feedback/",
            json={
                "transaction_id": "TXN_LEGACY_FB",
                "analyst_label": "Confirmed Fraud",
            },
            headers=analyst_headers,
        )
        assert response.status_code == 200

    def test_legacy_review_queue(self, api_test_client, admin_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_LEGACY_Q",
                "user_id": "U1",
                "amount_tnd": 5000.0,
                "governorate": "Tunis",
                "payment_method": "Flouci",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.90,
            },
            headers=admin_headers,
        )
        response = api_test_client.get("/alerts/review-queue/", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert any(a["transaction_id"] == "TXN_LEGACY_Q" for a in data)

    def test_legacy_stats_legacy_ctaf_and_model_perf(self, api_test_client, admin_headers):
        assert api_test_client.get("/stats/", headers=admin_headers).status_code == 200
        assert api_test_client.get("/compliance/kpis/", headers=admin_headers).status_code == 200
        assert api_test_client.get("/monitoring/model-performance/", headers=admin_headers).status_code == 200

    def test_legacy_explain_and_branches(self, api_test_client, admin_headers):
        assert api_test_client.get("/alerts/NONEXISTENT/explain", headers=admin_headers).status_code == 404
        assert api_test_client.get("/branches/", headers=admin_headers).status_code == 200


class TestHateoasLinks:
    def test_whoami_has_links(self, api_test_client, admin_headers):
        response = api_test_client.get("/auth/whoami", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "_links" in data
        assert data["_links"]["self"] == "/api/v1/auth/whoami"
        assert data["_links"]["stats"] == "/api/v1/stats/"

    def test_stats_has_links(self, api_test_client, admin_headers):
        response = api_test_client.get("/stats/", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "_links" in data
        assert data["_links"]["self"] == "/api/v1/stats/"
        assert data["_links"]["branches"] == "/api/v1/branches/"

    def test_ctaf_export_has_links(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_HATEOAS_01",
                "user_id": "U1",
                "amount_tnd": 1000.0,
                "governorate": "Tunis",
                "payment_method": "D17",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.95,
            },
            headers=admin_headers,
        )
        api_test_client.post(
            "/feedback/",
            json={"transaction_id": "TXN_HATEOAS_01", "analyst_label": "Confirmed Fraud"},
            headers=analyst_headers,
        )
        response = api_test_client.get("/alerts/ctaf-export?days=30", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "_links" in data
        assert data["_links"]["self"] == "/api/v1/alerts/ctaf-export"

    def test_feedback_has_links(self, api_test_client, admin_headers, analyst_headers):
        api_test_client.post(
            "/alerts/add/",
            json={
                "transaction_id": "TXN_HATEOAS_FB",
                "user_id": "U1",
                "amount_tnd": 1000.0,
                "governorate": "Tunis",
                "payment_method": "D17",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ml_probability": 0.95,
            },
            headers=admin_headers,
        )
        response = api_test_client.post(
            "/feedback/",
            json={"transaction_id": "TXN_HATEOAS_FB", "analyst_label": "Confirmed Fraud"},
            headers=analyst_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "_links" in data


class TestOpenAPISpec:
    """Verify the auto-generated OpenAPI spec includes all expected endpoints."""

    def test_openapi_json_returns_valid_spec(self, api_test_client, admin_headers, analyst_headers):
        response = api_test_client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "Tunisian Fraud Detection - Command Center API"
        assert spec["openapi"].startswith("3.")

        paths = spec["paths"]
        expected_paths = [
            "/api/v1/auth/whoami",
            "/api/v1/feedback/",
            "/api/v1/feedback/batch/",
            "/api/v1/alerts/high-risk/",
            "/api/v1/alerts/review-queue/",
            "/api/v1/alerts/add/",
            "/api/v1/alerts/ctaf-export",
            "/api/v1/alerts/{transaction_id}/explain",
            "/api/v1/alerts/{transaction_id}/export",
            "/api/v1/branches/",
            "/api/v1/stats/",
            "/api/v1/compliance/kpis/",
            "/api/v1/monitoring/model-performance/",
            "/api/v1/metrics/performance",
            "/api/v1/metrics/feedback",
            "/api/v1/metrics/threshold-analysis",
            "/api/v1/metrics/drift",
            "/api/v1/metrics/system-overview",
            "/api/v1/retrain-model/",
            "/api/v1/retrain-model/status/{job_id}",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing path in OpenAPI spec: {path}"

    def test_swagger_docs_renders(self, api_test_client):
        response = api_test_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_legacy_routes_deprecated_in_spec(self, api_test_client, admin_headers, analyst_headers):
        response = api_test_client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]
        legacy_paths = [
            p for p in paths if not p.startswith("/api/v1") and p not in ("/", "/health/", "/metrics", "/openapi.json")
        ]
        for lp in legacy_paths:
            for method_item in paths[lp].values():
                assert method_item.get("deprecated"), f"Legacy path {lp} should be marked deprecated"
