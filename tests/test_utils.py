"""Tests for shared utilities."""
import os
import sqlite3
import pytest
from shared.utils import (
    get_api_url, get_api_headers, get_sqlite_connection,
    ensure_dlq_table, log_failed_alert, update_dlq_status, retry_failed_alerts
)


class TestApiHelpers:
    def test_get_api_url_default(self, monkeypatch):
        monkeypatch.delenv("COMMAND_CENTER_API_URL", raising=False)
        monkeypatch.delenv("COMMAND_CENTER_API_VERSION", raising=False)
        assert get_api_url("alerts/") == "http://localhost:8001/api/v1/alerts/"

    def test_get_api_url_strips_leading_slash(self, monkeypatch):
        monkeypatch.delenv("COMMAND_CENTER_API_URL", raising=False)
        monkeypatch.delenv("COMMAND_CENTER_API_VERSION", raising=False)
        assert get_api_url("/alerts/") == "http://localhost:8001/api/v1/alerts/"

    def test_get_api_url_custom(self, monkeypatch):
        monkeypatch.setenv("COMMAND_CENTER_API_URL", "http://api:9000")
        monkeypatch.delenv("COMMAND_CENTER_API_VERSION", raising=False)
        assert get_api_url("health/") == "http://api:9000/api/v1/health/"

    def test_get_api_url_legacy_prefix_disabled(self, monkeypatch):
        monkeypatch.setenv("COMMAND_CENTER_API_URL", "http://api:9000")
        monkeypatch.setenv("COMMAND_CENTER_API_VERSION", "")
        assert get_api_url("health/") == "http://api:9000/health/"

    def test_get_api_headers_with_token(self, monkeypatch):
        monkeypatch.setenv("COMMAND_CENTER_API_TOKEN", "my_token")
        headers = get_api_headers()
        assert headers["Authorization"] == "Bearer my_token"
        assert headers["Content-Type"] == "application/json"

    def test_get_api_headers_no_token(self, monkeypatch):
        monkeypatch.delenv("COMMAND_CENTER_API_TOKEN", raising=False)
        headers = get_api_headers()
        assert "Authorization" not in headers


class TestSqliteConnection:
    def test_creates_connection(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = get_sqlite_connection(db_path)
        assert conn is not None
        conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = get_sqlite_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"
        conn.close()


class TestDeadLetterQueue:
    def test_ensure_dlq_table_creates_table(self, tmp_path):
        db_path = str(tmp_path / "dlq.db")
        conn = get_sqlite_connection(db_path)
        ensure_dlq_table(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failed_alerts'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_log_failed_alert(self, tmp_path, monkeypatch, sample_transaction_dict):
        db_path = str(tmp_path / "dlq.db")
        monkeypatch.setattr("shared.utils.DLQ_DB_PATH", db_path)

        log_failed_alert(sample_transaction_dict, {}, "TEST_ERROR", "Test error message")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT transaction_id, error_code FROM failed_alerts")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "TXN_TEST_001"
        assert row[1] == "TEST_ERROR"

    def test_update_dlq_status(self, tmp_path, monkeypatch, sample_transaction_dict):
        db_path = str(tmp_path / "dlq.db")
        monkeypatch.setattr("shared.utils.DLQ_DB_PATH", db_path)

        log_failed_alert(sample_transaction_dict, {}, "ERR", "msg")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM failed_alerts LIMIT 1")
        record_id = cursor.fetchone()[0]
        conn.close()

        update_dlq_status(record_id, "SUCCESS")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM failed_alerts WHERE id = ?", (record_id,))
        assert cursor.fetchone()[0] == "SUCCESS"
        conn.close()

    def test_retry_failed_alerts_does_not_copy_error_to_sar_report(self, tmp_path, monkeypatch, sample_transaction_dict):
        db_path = str(tmp_path / "dlq.db")
        monkeypatch.setattr("shared.utils.DLQ_DB_PATH", db_path)
        captured_payloads = []

        def capture_request(method, endpoint, payload=None, timeout=10):
            captured_payloads.append(payload)

            class Response:
                status_code = 200

            return Response()

        monkeypatch.setattr("shared.utils.make_authenticated_request", capture_request)
        log_failed_alert(sample_transaction_dict, {}, "ERR", "Original API error")

        retry_failed_alerts(max_attempts=3)

        assert captured_payloads
        assert captured_payloads[0]["sar_report"] is None

    def test_retry_failed_alerts_missing_db_is_quiet(self, tmp_path, monkeypatch, capsys):
        db_path = str(tmp_path / "missing_dlq.db")
        monkeypatch.setattr("shared.utils.DLQ_DB_PATH", db_path)

        retry_failed_alerts(max_attempts=3)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
