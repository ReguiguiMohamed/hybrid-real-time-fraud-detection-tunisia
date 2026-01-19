"""
Shared utilities for the fraud detection system
"""

import os
import requests
import sqlite3
from datetime import datetime


def get_api_headers():
    """Get headers with authentication token for API requests"""
    api_token = os.getenv("COMMAND_CENTER_API_TOKEN")
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    return headers


def get_api_url(endpoint=""):
    """Get the API URL with proper service discovery"""
    base_url = os.getenv("COMMAND_CENTER_API_URL", "http://localhost:8001")
    if endpoint.startswith('/'):
        endpoint = endpoint[1:]  # Remove leading slash if present
    return f"{base_url}/{endpoint}"


def make_authenticated_request(method, endpoint, payload=None, timeout=10):
    """Make an authenticated request to the API"""
    url = get_api_url(endpoint)
    headers = get_api_headers()
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, json=payload, headers=headers, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        return None


def log_failed_alert(transaction_data, alert_payload, error_code, error_message):
    """Log failed alerts to a dead letter queue for later processing"""
    try:
        # Create dead letter database if it doesn't exist
        db_path = "./data/dead_letter_queue.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failed_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                user_id TEXT,
                amount_tnd REAL,
                governorate TEXT,
                payment_method TEXT,
                timestamp TEXT,
                ml_probability REAL,
                error_code TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'PENDING'  -- PENDING, RETRYING, FAILED
            )
        """)

        # Insert failed alert
        cursor.execute("""
            INSERT INTO failed_alerts
            (transaction_id, user_id, amount_tnd, governorate, payment_method,
             timestamp, ml_probability, error_code, error_message, last_attempt, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_data.get('transaction_id'),
            transaction_data.get('user_id'),
            transaction_data.get('amount_tnd'),
            transaction_data.get('governorate'),
            transaction_data.get('payment_method'),
            transaction_data.get('timestamp'),
            transaction_data.get('ml_probability', 0.0),
            error_code,
            error_message,
            datetime.now().isoformat(),
            'PENDING'
        ))

        conn.commit()
        conn.close()

        print(f"Failed alert logged to dead letter queue: {transaction_data.get('transaction_id')}")
    except Exception as e:
        print(f"Error logging failed alert to dead letter queue: {e}")


def retry_failed_alerts(max_attempts=3):
    """Retry failed alerts from the dead letter queue"""
    try:
        db_path = "./data/dead_letter_queue.db"
        if not os.path.exists(db_path):
            print("No dead letter queue database found.")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get failed alerts that haven't exceeded max attempts
        cursor.execute("""
            SELECT id, transaction_id, user_id, amount_tnd, governorate, payment_method,
                   timestamp, ml_probability, error_code, error_message
            FROM failed_alerts
            WHERE status IN ('PENDING', 'RETRYING') AND attempts < ?
            ORDER BY created_at ASC
        """, (max_attempts,))

        failed_records = cursor.fetchall()
        conn.close()

        if not failed_records:
            print("No failed alerts to retry.")
            return

        print(f"Retrying {len(failed_records)} failed alerts...")

        for record in failed_records:
            record_id, transaction_id, user_id, amount_tnd, governorate, payment_method, \
            timestamp, ml_probability, error_code, error_message = record

            # Construct alert payload
            alert_payload = {
                "transaction_id": transaction_id,
                "user_id": user_id,
                "amount_tnd": amount_tnd,
                "governorate": governorate,
                "payment_method": payment_method,
                "timestamp": timestamp,
                "ml_probability": ml_probability,
                "sar_report": error_message  # Using error message as placeholder
            }

            # Attempt to resend the alert
            try:
                api_response = make_authenticated_request(
                    "POST",
                    "/alerts/add/",
                    payload=alert_payload,
                    timeout=10
                )

                if api_response and api_response.status_code == 200:
                    # Update status to SUCCESS
                    update_dlq_status(record_id, "SUCCESS")
                    print(f"Successfully resent alert for transaction: {transaction_id}")
                else:
                    # Increment attempt count and update status
                    increment_dlq_attempts(record_id, datetime.now().isoformat())
                    if api_response:
                        print(f"Failed to resend alert for {transaction_id}: {api_response.status_code}")
                    else:
                        print(f"Failed to resend alert for {transaction_id}: No response")
            except Exception as e:
                # Increment attempt count and update status
                increment_dlq_attempts(record_id, datetime.now().isoformat())
                print(f"Exception resending alert for {transaction_id}: {e}")

    except Exception as e:
        print(f"Error in retry_failed_alerts: {e}")


def update_dlq_status(record_id, status):
    """Update the status of a record in the dead letter queue"""
    try:
        db_path = "./data/dead_letter_queue.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE failed_alerts
            SET status = ?, last_attempt = ?
            WHERE id = ?
        """, (status, datetime.now().isoformat(), record_id))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating DLQ status: {e}")


def increment_dlq_attempts(record_id, last_attempt_time):
    """Increment the attempt count for a record in the dead letter queue"""
    try:
        db_path = "./data/dead_letter_queue.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE failed_alerts
            SET attempts = attempts + 1, last_attempt = ?
            WHERE id = ?
        """, (last_attempt_time, record_id))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error incrementing DLQ attempts: {e}")