"""Tests for the FastAPI Command Center API."""
import pytest
import sqlite3
from datetime import datetime, timezone


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


class TestExplainEndpoint:
    def test_explain_nonexistent_transaction(self, api_test_client, admin_headers):
        response = api_test_client.get("/alerts/NONEXISTENT/explain", headers=admin_headers)
        assert response.status_code == 404
