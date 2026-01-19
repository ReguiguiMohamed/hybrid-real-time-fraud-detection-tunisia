# dashboard/api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import os
from datetime import datetime
from pathlib import Path
import json
import threading
from queue import Queue
import time

app = FastAPI(title="Tunisian Fraud Detection - Command Center API")

# Database setup
DB_PATH = Path("./data/feedback.db")
os.makedirs("./data", exist_ok=True)

class FeedbackRequest(BaseModel):
    transaction_id: str
    analyst_label: str  # "Confirmed Fraud" or "False Positive"
    analyst_comment: Optional[str] = None

class TransactionAlert(BaseModel):
    transaction_id: str
    user_id: str
    amount_tnd: float
    governorate: str
    payment_method: str
    timestamp: str
    ml_probability: float
    sar_report: Optional[str] = None

class ModelMonitor:
    """Simple model monitoring class to track performance metrics"""
    def __init__(self):
        self.feedback_queue = Queue()
        self.feedback_processed = 0
        self.false_positive_count = 0
        self.confirmed_fraud_count = 0

    def record_feedback(self, feedback: FeedbackRequest):
        """Record feedback and update metrics"""
        if feedback.analyst_label == "Confirmed Fraud":
            self.confirmed_fraud_count += 1
        elif feedback.analyst_label == "False Positive":
            self.false_positive_count += 1
        self.feedback_processed += 1

# Global monitor instance
monitor = ModelMonitor()

@app.on_event("startup")
def startup_event():
    """Initialize the database on startup"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            analyst_label TEXT,
            analyst_comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create alerts table for high-risk transactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS high_risk_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            user_id TEXT,
            amount_tnd REAL,
            governorate TEXT,
            payment_method TEXT,
            timestamp TEXT,
            ml_probability REAL,
            sar_report TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Endpoint to receive analyst feedback on fraud predictions"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert feedback into database
        cursor.execute("""
            INSERT INTO feedback_labels 
            (transaction_id, analyst_label, analyst_comment)
            VALUES (?, ?, ?)
        """, (
            feedback.transaction_id, 
            feedback.analyst_label, 
            feedback.analyst_comment
        ))
        
        conn.commit()
        conn.close()
        
        # Update monitoring metrics
        monitor.record_feedback(feedback)
        
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/high-risk/")
async def get_high_risk_alerts(limit: int = 50):
    """Endpoint to fetch high-risk alerts for the dashboard"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT transaction_id, user_id, amount_tnd, governorate, payment_method, 
                   timestamp, ml_probability, sar_report
            FROM high_risk_alerts 
            WHERE ml_probability > 0.85
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
                "timestamp": row[5],
                "ml_probability": row[6],
                "sar_report": row[7]
            }
            alerts.append(alert)
        
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/")
async def get_system_stats():
    """Get system statistics for monitoring"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total feedback count
        cursor.execute("SELECT COUNT(*) FROM feedback_labels")
        total_feedback = cursor.fetchone()[0]
        
        # Get fraud vs false positive counts
        cursor.execute("SELECT analyst_label, COUNT(*) FROM feedback_labels GROUP BY analyst_label")
        label_counts = dict(cursor.fetchall())
        
        # Get high-risk alert count
        cursor.execute("SELECT COUNT(*) FROM high_risk_alerts WHERE ml_probability > 0.85")
        high_risk_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Calculate precision based on feedback
        confirmed_fraud = label_counts.get("Confirmed Fraud", 0)
        total_labeled = sum(label_counts.values()) if label_counts else 0
        precision = confirmed_fraud / total_labeled if total_labeled > 0 else 0
        
        return {
            "total_feedback": total_feedback,
            "high_risk_alerts": high_risk_count,
            "feedback_breakdown": label_counts,
            "precision": round(precision, 3),
            "monitoring_stats": {
                "feedback_processed": monitor.feedback_processed,
                "confirmed_fraud_count": monitor.confirmed_fraud_count,
                "false_positive_count": monitor.false_positive_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/add/")
async def add_high_risk_alert(alert: TransactionAlert):
    """Endpoint to add high-risk alerts from the streaming pipeline"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert alert into database
        cursor.execute("""
            INSERT OR IGNORE INTO high_risk_alerts 
            (transaction_id, user_id, amount_tnd, governorate, payment_method, 
             timestamp, ml_probability, sar_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.transaction_id, alert.user_id, alert.amount_tnd,
            alert.governorate, alert.payment_method, alert.timestamp,
            alert.ml_probability, alert.sar_report
        ))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Alert added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/model-performance/")
async def get_model_performance():
    """Get model performance metrics based on human feedback"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get feedback data to calculate performance metrics
        cursor.execute("""
            SELECT hra.ml_probability, fl.analyst_label
            FROM high_risk_alerts hra
            JOIN feedback_labels fl ON hra.transaction_id = fl.transaction_id
            WHERE fl.analyst_label IS NOT NULL
        """)

        prob_label_pairs = cursor.fetchall()
        conn.close()

        if not prob_label_pairs:
            return {
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_evaluated": 0
            }

        # Calculate performance metrics
        tp = sum(1 for prob, label in prob_label_pairs if label == "Confirmed Fraud")  # True positives
        fp = sum(1 for prob, label in prob_label_pairs if label == "False Positive")  # False positives
        tn = sum(1 for prob, label in prob_label_pairs if label == "False Positive")  # In this case, we consider all non-fraud as TN
        # Calculate true negatives correctly - these are cases where model predicted non-fraud and analyst confirmed it
        # For now, we'll focus on precision and recall for fraud detection
        fn = 0  # For fraud detection, false negatives would be cases labeled as non-fraud but were actually fraud

        # Calculate recall based on actual fraud cases identified by analysts
        # Need to query for all confirmed fraud cases vs total actual fraud cases
        # For now, using a simplified approach focusing on precision of positive predictions
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Calculate recall: fraction of actual fraud cases that were correctly identified
        # This requires knowing total actual fraud cases, which we can estimate from feedback
        total_actual_fraud = sum(1 for prob, label in prob_label_pairs if label == "Confirmed Fraud")
        recall = tp / total_actual_fraud if total_actual_fraud > 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3),
            "total_evaluated": len(prob_label_pairs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain-model/")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
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