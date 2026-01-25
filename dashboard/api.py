# dashboard/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
import hashlib
import logging
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared.utils import get_sqlite_connection, retry_failed_alerts
from monitoring import ForensicAnalyticEngine

app = FastAPI(title="Tunisian Fraud Detection - Command Center API")

# Authentication setup
security = HTTPBearer()

# Load API tokens from environment variables
ANALYST_TOKEN = os.getenv("ANALYST_TOKEN")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or os.getenv("API_TOKEN")

if not ANALYST_TOKEN:
    print("WARNING: ANALYST_TOKEN not set. Using default token for development.")
    ANALYST_TOKEN = "default_analyst_token"
if not ADMIN_TOKEN:
    print("WARNING: ADMIN_TOKEN not set. Using default token for development.")
    ADMIN_TOKEN = "default_admin_token"

ANALYST_TOKEN_HASH = hashlib.sha256(ANALYST_TOKEN.encode()).hexdigest()
ADMIN_TOKEN_HASH = hashlib.sha256(ADMIN_TOKEN.encode()).hexdigest()

def require_scopes(scopes):
    """Verify the API token against required scopes."""
    def verifier(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        if token_hash == ADMIN_TOKEN_HASH:
            role = "admin"
        elif token_hash == ANALYST_TOKEN_HASH:
            role = "analyst"
        else:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if role not in scopes:
            raise HTTPException(status_code=403, detail="Forbidden")
        return {"role": role}
    return verifier

# Database setup
DB_PATH = Path("./data/feedback.db")
os.makedirs("./data", exist_ok=True)

def parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def parse_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

RANDOM_SAMPLE_RATE = max(0.0, min(parse_float_env("RANDOM_SAMPLE_RATE", 0.01), 1.0))

FEATURE_LABELS = {
    "v_count": "High velocity (v_count)",
    "g_dist": "Multi-governorate travel (g_dist)",
    "avg_amount": "High value transfer (avg_amount)",
    "is_smurfing": "Structuring pattern (is_smurfing)",
    "high_velocity_flag": "D17 velocity cap (high_velocity_flag)",
    "velocity_risk": "Velocity risk flag",
    "travel_risk": "Travel risk flag",
    "high_value_risk": "High value risk flag",
    "d17_risk": "D17 Flouci risk flag",
    "risk_score": "Composite risk score"
}

def get_champion_model_path():
    conn = get_sqlite_connection(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_path
        FROM model_registry
        WHERE is_champion = 1
        ORDER BY promoted_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

class FeedbackRequest(BaseModel):
    transaction_id: str
    analyst_label: str  # "Confirmed Fraud" or "False Positive"
    analyst_comment: Optional[str] = None
    branch_id: Optional[str] = None

def log_audit_event(entity_type, entity_id, action, user_id, previous_state, new_state):
    conn = get_sqlite_connection(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_logs
        (entity_type, entity_id, action, user_id, previous_state, new_state)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        entity_type,
        entity_id,
        action,
        user_id,
        previous_state,
        new_state
    ))
    conn.commit()
    conn.close()

class TransactionAlert(BaseModel):
    transaction_id: str
    user_id: str
    amount_tnd: float
    governorate: str
    payment_method: str
    branch_id: Optional[str] = None
    timestamp: str
    ml_probability: float
    sar_report: Optional[str] = None
    alert_type: Optional[str] = "high_risk"

def parse_feature_importance(feature_payload, limit=3):
    if not feature_payload:
        return []
    try:
        feature_items = json.loads(feature_payload)
    except json.JSONDecodeError:
        return []

    normalized = []
    for item in feature_items:
        if isinstance(item, dict):
            feature_name = item.get("feature")
            score = item.get("score")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            feature_name, score = item
        else:
            continue
        if feature_name is None:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None
        normalized.append((feature_name, score_value))

    normalized.sort(key=lambda item: item[1] if item[1] is not None else 0, reverse=True)
    top_features = normalized[:limit]
    return [
        {
            "feature": feature_name,
            "description": FEATURE_LABELS.get(feature_name, feature_name),
            "score": round(score, 4) if score is not None else None
        }
        for feature_name, score in top_features
    ]

monitoring_engine = ForensicAnalyticEngine(DB_PATH)

@app.on_event("startup")
def startup_event():
    """Initialize the database on startup"""
    conn = get_sqlite_connection(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            analyst_label TEXT,
            analyst_comment TEXT,
            branch_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create alerts table for review queue entries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS high_risk_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            user_id TEXT,
            amount_tnd REAL,
            governorate TEXT,
            payment_method TEXT,
            branch_id TEXT,
            timestamp TEXT,
            ml_probability REAL,
            sar_report TEXT,
            alert_type TEXT DEFAULT 'high_risk',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("PRAGMA table_info(feedback_labels)")
    feedback_columns = {row[1] for row in cursor.fetchall()}
    if "branch_id" not in feedback_columns:
        cursor.execute("ALTER TABLE feedback_labels ADD COLUMN branch_id TEXT")


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            version_id TEXT PRIMARY KEY,
            model_path TEXT NOT NULL,
            f1_score REAL,
            auc REAL,
            is_champion INTEGER DEFAULT 0,
            promoted_at DATETIME,
            training_samples_count INTEGER,
            feature_importance TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_id TEXT,
            action TEXT,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            previous_state TEXT,
            new_state TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(model_registry)")
    registry_columns = {row[1] for row in cursor.fetchall()}
    if "feature_importance" not in registry_columns:
        cursor.execute("ALTER TABLE model_registry ADD COLUMN feature_importance TEXT")


    cursor.execute("PRAGMA table_info(high_risk_alerts)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "alert_type" not in existing_columns:
        cursor.execute("ALTER TABLE high_risk_alerts ADD COLUMN alert_type TEXT DEFAULT 'high_risk'")
        cursor.execute("UPDATE high_risk_alerts SET alert_type = 'high_risk' WHERE alert_type IS NULL")
    if "branch_id" not in existing_columns:
        cursor.execute("ALTER TABLE high_risk_alerts ADD COLUMN branch_id TEXT")

    conn.commit()
    conn.close()
    start_dlq_retry_worker()

@app.on_event("shutdown")
def shutdown_event():
    stop_event = getattr(app.state, "dlq_retry_stop", None)
    if stop_event:
        stop_event.set()
    thread = getattr(app.state, "dlq_retry_thread", None)
    if thread and thread.is_alive():
        thread.join(timeout=2)

def start_dlq_retry_worker():
    if getattr(app.state, "dlq_retry_thread", None) and app.state.dlq_retry_thread.is_alive():
        return

    interval = max(1, parse_int_env("DLQ_RETRY_INTERVAL_SECONDS", 60))
    max_attempts = max(1, parse_int_env("DLQ_RETRY_MAX_ATTEMPTS", 3))
    stop_event = threading.Event()
    app.state.dlq_retry_stop = stop_event

    def retry_loop():
        while not stop_event.is_set():
            try:
                retry_failed_alerts(max_attempts=max_attempts)
            except Exception:
                logging.exception("DLQ retry worker error")
            stop_event.wait(interval)

    thread = threading.Thread(target=retry_loop, name="dlq-retry-worker", daemon=True)
    app.state.dlq_retry_thread = thread
    thread.start()

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest, user_id: Optional[str] = Header(None), auth=Depends(require_scopes({"analyst", "admin"}))):
    """Endpoint to receive analyst feedback on fraud predictions"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        branch_id = feedback.branch_id
        if not branch_id:
            cursor.execute(
                "SELECT branch_id FROM high_risk_alerts WHERE transaction_id = ?",
                (feedback.transaction_id,)
            )
            row = cursor.fetchone()
            branch_id = row[0] if row else None

        cursor.execute("""
            SELECT analyst_label, analyst_comment
            FROM feedback_labels
            WHERE transaction_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (feedback.transaction_id,))
        previous_feedback = cursor.fetchone()

        # Insert feedback into database
        cursor.execute("""
            INSERT INTO feedback_labels
            (transaction_id, analyst_label, analyst_comment, branch_id)
            VALUES (?, ?, ?, ?)
        """, (
            feedback.transaction_id,
            feedback.analyst_label,
            feedback.analyst_comment,
            branch_id
        ))

        conn.commit()
        conn.close()

        previous_state = None
        if previous_feedback:
            previous_state = json.dumps({
                "analyst_label": previous_feedback[0],
                "analyst_comment": previous_feedback[1]
            })
        new_state = json.dumps({
            "analyst_label": feedback.analyst_label,
            "analyst_comment": feedback.analyst_comment,
            "branch_id": branch_id
        })
        log_audit_event(
            "ALERT",
            feedback.transaction_id,
            "CLASSIFY",
            user_id or "unknown",
            previous_state,
            new_state
        )

        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/high-risk/")
async def get_high_risk_alerts(limit: int = 50, branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Endpoint to fetch high-risk alerts for the dashboard"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        if branch_id:
            cursor.execute("""
                SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                       branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk')
                FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'high_risk'
                  AND ml_probability > 0.85
                  AND branch_id = ?
                ORDER BY ml_probability DESC
                LIMIT ?
            """, (branch_id, limit))
        else:
            cursor.execute("""
                SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                       branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk')
                FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'high_risk'
                  AND ml_probability > 0.85
                ORDER BY ml_probability DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert = {
                "transaction_id": row[0],
                "user_id": row[1],
                "amount_tnd": row[2],
                "governorate": row[3],
                "payment_method": row[4],
                "branch_id": row[5],
                "timestamp": row[6],
                "ml_probability": row[7],
                "sar_report": row[8],
                "alert_type": row[9]
            }
            alerts.append(alert)

        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/review-queue/")
async def get_review_queue(limit: int = 100, alert_type: Optional[str] = None, branch_id: Optional[str] = None,
                           auth=Depends(require_scopes({"analyst", "admin"}))):
    """Endpoint to fetch review queue alerts, including random samples"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        if alert_type:
            cursor.execute("""
                SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                       branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk')
                FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = ?
                  AND (? IS NULL OR branch_id = ?)
                ORDER BY created_at DESC
                LIMIT ?
            """, (alert_type, branch_id, branch_id, limit))
        else:
            cursor.execute("""
                SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                       branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk')
                FROM high_risk_alerts
                WHERE (? IS NULL OR branch_id = ?)
                ORDER BY created_at DESC
                LIMIT ?
            """, (branch_id, branch_id, limit))

        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert = {
                "transaction_id": row[0],
                "user_id": row[1],
                "amount_tnd": row[2],
                "governorate": row[3],
                "payment_method": row[4],
                "branch_id": row[5],
                "timestamp": row[6],
                "ml_probability": row[7],
                "sar_report": row[8],
                "alert_type": row[9]
            }
            alerts.append(alert)

        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/branches/")
async def list_branches(auth=Depends(require_scopes({"analyst", "admin"}))):
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT branch_id
            FROM high_risk_alerts
            WHERE branch_id IS NOT NULL AND branch_id != ''
            ORDER BY branch_id
        """)
        branches = [row[0] for row in cursor.fetchall()]
        conn.close()
        return branches
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/")
async def get_system_stats(branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Get system statistics for monitoring"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        # Get total feedback count
        if branch_id:
            cursor.execute("SELECT COUNT(*) FROM feedback_labels WHERE branch_id = ?", (branch_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM feedback_labels")
        total_feedback = cursor.fetchone()[0]

        # Get fraud vs false positive counts
        if branch_id:
            cursor.execute(
                "SELECT analyst_label, COUNT(*) FROM feedback_labels WHERE branch_id = ? GROUP BY analyst_label",
                (branch_id,)
            )
        else:
            cursor.execute("SELECT analyst_label, COUNT(*) FROM feedback_labels GROUP BY analyst_label")
        label_counts = dict(cursor.fetchall())

        if branch_id:
            cursor.execute("""
                SELECT COALESCE(hra.alert_type, 'unknown') AS alert_type,
                       fl.analyst_label,
                       COUNT(*)
                FROM feedback_labels fl
                LEFT JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                  AND fl.branch_id = ?
                GROUP BY alert_type, fl.analyst_label
            """, (branch_id,))
        else:
            cursor.execute("""
                SELECT COALESCE(hra.alert_type, 'unknown') AS alert_type,
                       fl.analyst_label,
                       COUNT(*)
                FROM feedback_labels fl
                LEFT JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                GROUP BY alert_type, fl.analyst_label
            """)
        label_counts_by_type = {}
        for alert_type, analyst_label, count in cursor.fetchall():
            label_counts_by_type.setdefault(alert_type, {})[analyst_label] = count

        # Get high-risk alert count
        if branch_id:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'high_risk'
                  AND ml_probability > 0.85
                  AND branch_id = ?
            """, (branch_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'high_risk'
                  AND ml_probability > 0.85
            """)
        high_risk_count = cursor.fetchone()[0]

        if branch_id:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'random_sample'
                  AND branch_id = ?
            """, (branch_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'random_sample'
            """)
        random_sample_count = cursor.fetchone()[0]

        if branch_id:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'uncertainty_sample'
                  AND branch_id = ?
            """, (branch_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM high_risk_alerts
                WHERE COALESCE(alert_type, 'high_risk') = 'uncertainty_sample'
            """)
        uncertainty_sample_count = cursor.fetchone()[0]

        if branch_id:
            cursor.execute("SELECT COUNT(*) FROM high_risk_alerts WHERE branch_id = ?", (branch_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM high_risk_alerts")
        review_queue_total = cursor.fetchone()[0]

        conn.close()

        # Calculate precision based on high-risk alerts only
        high_risk_counts = label_counts_by_type.get("high_risk", {})
        confirmed_fraud = high_risk_counts.get("Confirmed Fraud", 0)
        false_positive = high_risk_counts.get("False Positive", 0)
        high_risk_precision = confirmed_fraud / (confirmed_fraud + false_positive) if (confirmed_fraud + false_positive) > 0 else 0

        random_sample_counts = label_counts_by_type.get("random_sample", {})
        random_sample_fraud = random_sample_counts.get("Confirmed Fraud", 0)
        random_sample_non_fraud = random_sample_counts.get("False Positive", 0)
        random_sample_fraud_rate = (
            random_sample_fraud / (random_sample_fraud + random_sample_non_fraud)
            if (random_sample_fraud + random_sample_non_fraud) > 0 else 0
        )

        return {
            "total_feedback": total_feedback,
            "high_risk_alerts": high_risk_count,
            "random_sample_alerts": random_sample_count,
            "uncertainty_sample_alerts": uncertainty_sample_count,
            "review_queue_total": review_queue_total,
            "random_sample_rate": RANDOM_SAMPLE_RATE,
            "feedback_breakdown": label_counts,
            "feedback_breakdown_by_type": label_counts_by_type,
            "precision": round(high_risk_precision, 3),
            "precision_scope": "high_risk_only",
            "high_risk_precision": round(high_risk_precision, 3),
            "random_sample_fraud_rate": round(random_sample_fraud_rate, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/add/")
async def add_high_risk_alert(alert: TransactionAlert, auth=Depends(require_scopes({"admin"}))):
    """Endpoint to add high-risk alerts from the streaming pipeline"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        alert_type = alert.alert_type or "high_risk"

        # Insert alert into database
        cursor.execute("""
            INSERT OR IGNORE INTO high_risk_alerts
            (transaction_id, user_id, amount_tnd, governorate, payment_method, branch_id,
             timestamp, ml_probability, sar_report, alert_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.transaction_id, alert.user_id, alert.amount_tnd,
            alert.governorate, alert.payment_method, alert.branch_id, alert.timestamp,
            alert.ml_probability, alert.sar_report, alert_type
        ))

        conn.commit()
        conn.close()

        return {"status": "success", "message": "Alert added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/model-performance/")
async def get_model_performance(branch_id: Optional[str] = None, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Get model performance metrics based on human feedback"""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()

        # Get feedback data to calculate performance metrics
        # IMPORTANT: This represents performance on the subset of transactions that were reviewed by analysts
        # NOT the overall model performance across all transactions
        if branch_id:
            cursor.execute("""
                SELECT hra.ml_probability, COALESCE(hra.alert_type, 'high_risk'), fl.analyst_label
                FROM high_risk_alerts hra
                JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                  AND COALESCE(hra.alert_type, 'high_risk') IN ('high_risk', 'random_sample')
                  AND fl.branch_id = ?
            """, (branch_id,))
        else:
            cursor.execute("""
                SELECT hra.ml_probability, COALESCE(hra.alert_type, 'high_risk'), fl.analyst_label
                FROM high_risk_alerts hra
                JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label IS NOT NULL
                AND COALESCE(hra.alert_type, 'high_risk') IN ('high_risk', 'random_sample')
            """)
    


        prob_label_pairs = cursor.fetchall()
        conn.close()

        if not prob_label_pairs:
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_evaluated": 0,
                "note": "Metrics calculated only on reviewed alerts, not overall model performance",
                "warning": "Cannot calculate true model performance without sampling negative cases"
            }

        # Calculate performance metrics properly for the reviewed subset
        # High-risk alerts are model-flagged fraud; random samples are low-risk reviews
        tp = sum(1 for _, alert_type, label in prob_label_pairs
                 if alert_type == "high_risk" and label == "Confirmed Fraud")
        fp = sum(1 for _, alert_type, label in prob_label_pairs
                 if alert_type == "high_risk" and label == "False Positive")
        tn = sum(1 for _, alert_type, label in prob_label_pairs
                 if alert_type == "random_sample" and label == "False Positive")
        fn_sampled = sum(1 for _, alert_type, label in prob_label_pairs
                         if alert_type == "random_sample" and label == "Confirmed Fraud")

        # Calculate precision based only on reviewed alerts where model predicted fraud
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate recall using sampled false negatives
        random_sample_reviewed = tn + fn_sampled
        if random_sample_reviewed == 0:
            reviewed_recall = 0
            estimated_fn = 0
            estimated_recall = 0
        else:
            reviewed_recall = tp / (tp + fn_sampled) if (tp + fn_sampled) > 0 else 0
            estimated_fn = fn_sampled / RANDOM_SAMPLE_RATE if RANDOM_SAMPLE_RATE > 0 else 0
            estimated_recall = tp / (tp + estimated_fn) if (tp + estimated_fn) > 0 else 0

        f1_score = 2 * (precision * estimated_recall) / (precision + estimated_recall) if (precision + estimated_recall) > 0 else 0

        return {
            "precision": round(precision, 3),
            "recall": round(estimated_recall, 3),
            "reviewed_recall": round(reviewed_recall, 3),
            "f1_score": round(f1_score, 3),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn_sampled,
            "estimated_false_negatives": round(estimated_fn, 3),
            "random_sample_reviewed": random_sample_reviewed,
            "random_sample_rate": RANDOM_SAMPLE_RATE,
            "total_evaluated": len(prob_label_pairs),
            "note": "Metrics combine high-risk reviews with random-sample reviews to estimate recall.",
            "warning": "Estimated recall assumes random samples represent low-risk traffic.",
            "interpretation": "Precision reflects alert performance; recall is sampling-adjusted."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/{transaction_id}/explain")
async def explain_alert(transaction_id: str, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Explain the top risk factors for a transaction using model feature importance."""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, COALESCE(alert_type, 'high_risk')
            FROM high_risk_alerts
            WHERE transaction_id = ?
        """, (transaction_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Transaction not found")

        cursor.execute("""
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        registry_row = cursor.fetchone()
        conn.close()

        if not registry_row or not registry_row[0]:
            return {
                "transaction_id": transaction_id,
                "alert_type": row[1],
                "top_risk_factors": [],
                "note": "No champion feature importance registered yet."
            }

        factors = parse_feature_importance(registry_row[0], limit=3)

        return {
            "transaction_id": transaction_id,
            "alert_type": row[1],
            "top_risk_factors": factors
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/{transaction_id}/export")
async def export_alert(transaction_id: str, auth=Depends(require_scopes({"analyst", "admin"}))):
    """Export a single alert for compliance filing."""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT transaction_id, user_id, amount_tnd, governorate, payment_method,
                   branch_id, timestamp, ml_probability, sar_report, COALESCE(alert_type, 'high_risk')
            FROM high_risk_alerts
            WHERE transaction_id = ?
        """, (transaction_id,))
        alert_row = cursor.fetchone()
        if not alert_row:
            conn.close()
            raise HTTPException(status_code=404, detail="Transaction not found")

        cursor.execute("""
            SELECT analyst_label, analyst_comment, timestamp
            FROM feedback_labels
            WHERE transaction_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (transaction_id,))
        feedback_row = cursor.fetchone()

        cursor.execute("""
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        registry_row = cursor.fetchone()
        conn.close()

        factors = parse_feature_importance(registry_row[0] if registry_row else None, limit=3)

        analyst_payload = None
        if feedback_row:
            analyst_payload = {
                "label": feedback_row[0],
                "comment": feedback_row[1],
                "timestamp": feedback_row[2]
            }

        return {
            "transaction_id": alert_row[0],
            "user_id": alert_row[1],
            "amount_tnd": alert_row[2],
            "governorate": alert_row[3],
            "payment_method": alert_row[4],
            "branch_id": alert_row[5],
            "timestamp": alert_row[6],
            "ml_probability": alert_row[7],
            "sar_report": alert_row[8],
            "alert_type": alert_row[9],
            "top_risk_factors": factors,
            "analyst_review": analyst_payload,
            "exported_at": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/ctaf-export")
async def export_ctaf(days: int = 7, branch_id: Optional[str] = None, auth=Depends(require_scopes({"admin"}))):
    """Export confirmed fraud alerts for CTAF reporting."""
    try:
        conn = get_sqlite_connection(str(DB_PATH))
        cursor = conn.cursor()
        cutoff = f"-{days} days"
        if branch_id:
            cursor.execute("""
                SELECT hra.transaction_id, hra.user_id, hra.amount_tnd, hra.governorate, hra.payment_method,
                       hra.branch_id, hra.timestamp, hra.ml_probability, hra.sar_report,
                       fl.analyst_label, fl.analyst_comment, fl.timestamp
                FROM feedback_labels fl
                JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label = 'Confirmed Fraud'
                  AND fl.timestamp >= datetime('now', ?)
                  AND fl.branch_id = ?
                ORDER BY fl.timestamp DESC
            """, (cutoff, branch_id))
        else:
            cursor.execute("""
                SELECT hra.transaction_id, hra.user_id, hra.amount_tnd, hra.governorate, hra.payment_method,
                       hra.branch_id, hra.timestamp, hra.ml_probability, hra.sar_report,
                       fl.analyst_label, fl.analyst_comment, fl.timestamp
                FROM feedback_labels fl
                JOIN high_risk_alerts hra ON hra.transaction_id = fl.transaction_id
                WHERE fl.analyst_label = 'Confirmed Fraud'
                  AND fl.timestamp >= datetime('now', ?)
                ORDER BY fl.timestamp DESC
            """, (cutoff,))

        rows = cursor.fetchall()

        cursor.execute("""
            SELECT feature_importance
            FROM model_registry
            WHERE is_champion = 1
            ORDER BY promoted_at DESC
            LIMIT 1
        """)
        registry_row = cursor.fetchone()
        conn.close()

        factors = parse_feature_importance(registry_row[0] if registry_row else None, limit=3)

        cases = []
        for row in rows:
            cases.append({
                "transaction_id": row[0],
                "user_id": row[1],
                "amount_tnd": row[2],
                "governorate": row[3],
                "payment_method": row[4],
                "branch_id": row[5],
                "timestamp": row[6],
                "ml_probability": row[7],
                "sar_report": row[8],
                "analyst_label": row[9],
                "analyst_comment": row[10],
                "analyst_timestamp": row[11],
                "top_risk_factors": factors
            })

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "days": days,
            "branch_id": branch_id,
            "total_cases": len(cases),
            "cases": cases
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/performance")
async def get_performance_metrics(auth=Depends(require_scopes({"analyst", "admin"}))):
    return monitoring_engine.get_performance_metrics()

@app.get("/metrics/feedback")
async def get_feedback_analysis(auth=Depends(require_scopes({"analyst", "admin"}))):
    return monitoring_engine.get_feedback_analysis()

@app.get("/metrics/threshold-analysis")
async def get_threshold_analysis(auth=Depends(require_scopes({"analyst", "admin"}))):
    return monitoring_engine.get_ml_threshold_analysis()

@app.get("/metrics/system-overview")
async def get_system_overview(auth=Depends(require_scopes({"analyst", "admin"}))):
    return {
        "performance": monitoring_engine.get_performance_metrics(),
        "feedback": monitoring_engine.get_feedback_analysis(),
        "threshold_recommendation": monitoring_engine.get_ml_threshold_analysis()
    }

@app.post("/retrain-model/")
async def trigger_model_retraining(background_tasks: BackgroundTasks, auth=Depends(require_scopes({"admin"}))):
    """Trigger model retraining based on accumulated feedback"""
    try:
        # Import here to avoid circular dependencies
        from src.ml.train_model import FraudModelTrainer

        def run_retraining():
            try:
                trainer = FraudModelTrainer()
                success = trainer.train_champion_challenger()
                if success:
                    print("✅ Model retrained successfully from API trigger")
                else:
                    print("ℹ️  No retraining needed based on current feedback")
            except Exception as e:
                print(f"Error during model retraining: {e}")

        # Run retraining in background to avoid blocking the API
        background_tasks.add_task(run_retraining)

        return {"status": "success", "message": "Model retraining triggered in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint for the API"""
    return {"status": "healthy", "service": "fraud-detection-command-center-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
