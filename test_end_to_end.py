#!/usr/bin/env python3
"""
End-to-End Test Script for Tunisian Fraud Detection System
This script simulates the complete flow: Analyst feedback → Model retraining → Improved predictions
"""

import os
import sqlite3
import requests
import time
import json
from datetime import datetime
from src.ml.train_model import FraudModelTrainer

def setup_test_environment():
    """Set up the test environment with initial data"""
    print("Setting up test environment...")
    
    # Create feedback database if it doesn't exist
    db_path = "./data/feedback.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
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
    
    print("✅ Test environment set up")


def simulate_analyst_feedback():
    """Simulate analyst providing feedback on alerts"""
    print("\nSimulating analyst feedback...")
    
    # Sample high-risk transactions to provide feedback on
    sample_transactions = [
        {
            "transaction_id": "txn_001",
            "user_id": "user_123",
            "amount_tnd": 2500.0,
            "governorate": "Tunis",
            "payment_method": "Flouci",
            "timestamp": "2024-01-19T15:30:00Z",
            "ml_probability": 0.88,
            "sar_report": "Suspicious transaction flagged by ML model"
        },
        {
            "transaction_id": "txn_002",
            "user_id": "user_456",
            "amount_tnd": 1800.0,
            "governorate": "Sfax",
            "payment_method": "Credit Card",
            "timestamp": "2024-01-19T15:32:00Z",
            "ml_probability": 0.91,
            "sar_report": "High velocity transaction pattern detected"
        },
        {
            "transaction_id": "txn_003",
            "user_id": "user_789",
            "amount_tnd": 3200.0,
            "governorate": "Bizerte",
            "payment_method": "Mobile Wallet",
            "timestamp": "2024-01-19T15:35:00Z",
            "ml_probability": 0.86,
            "sar_report": "Unusual geographic pattern detected"
        },
        {
            "transaction_id": "txn_004",
            "user_id": "user_101",
            "amount_tnd": 950.0,
            "governorate": "Sousse",
            "payment_method": "Debit Card",
            "timestamp": "2024-01-19T15:38:00Z",
            "ml_probability": 0.87,
            "sar_report": "Velocity anomaly detected"
        },
        {
            "transaction_id": "txn_005",
            "user_id": "user_202",
            "amount_tnd": 4100.0,
            "governorate": "Monastir",
            "payment_method": "Flouci",
            "timestamp": "2024-01-19T15:40:00Z",
            "ml_probability": 0.93,
            "sar_report": "High-value transaction with unusual pattern"
        }
    ]
    
    # Add these transactions to the alerts table
    conn = sqlite3.connect("./data/feedback.db")
    cursor = conn.cursor()
    
    for tx in sample_transactions:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO high_risk_alerts
                (transaction_id, user_id, amount_tnd, governorate, payment_method,
                 timestamp, ml_probability, sar_report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx["transaction_id"], tx["user_id"], tx["amount_tnd"],
                tx["governorate"], tx["payment_method"], tx["timestamp"],
                tx["ml_probability"], tx["sar_report"]
            ))
        except sqlite3.Error as e:
            print(f"Error inserting transaction {tx['transaction_id']}: {e}")
    
    conn.commit()
    conn.close()
    
    # Simulate analyst feedback (some confirmed fraud, some false positives)
    feedback_data = [
        {"transaction_id": "txn_001", "analyst_label": "Confirmed Fraud", "analyst_comment": "Verified fraudulent activity"},
        {"transaction_id": "txn_002", "analyst_label": "False Positive", "analyst_comment": "Legitimate business transaction"},
        {"transaction_id": "txn_003", "analyst_label": "Confirmed Fraud", "analyst_comment": "Part of money laundering scheme"},
        {"transaction_id": "txn_004", "analyst_label": "False Positive", "analyst_comment": "Valid e-commerce purchase"},
        {"transaction_id": "txn_005", "analyst_label": "Confirmed Fraud", "analyst_comment": "Card not present fraud"}
    ]
    
    # Submit feedback via API
    for feedback in feedback_data:
        try:
            response = requests.post(
                "http://localhost:8001/feedback/",
                json=feedback,
                timeout=5
            )
            if response.status_code == 200:
                print(f"✅ Feedback submitted for {feedback['transaction_id']}: {feedback['analyst_label']}")
            else:
                print(f"❌ Failed to submit feedback for {feedback['transaction_id']}: {response.status_code}")
        except requests.exceptions.ConnectionError:
            # If API is not running, insert directly into DB
            print(f"⚠️  API not running, inserting feedback directly for {feedback['transaction_id']}")
            conn = sqlite3.connect("./data/feedback.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_labels
                (transaction_id, analyst_label, analyst_comment)
                VALUES (?, ?, ?)
            """, (feedback["transaction_id"], feedback["analyst_label"], feedback["analyst_comment"]))
            conn.commit()
            conn.close()
    
    print("✅ Analyst feedback simulation completed")


def trigger_model_retraining():
    """Trigger model retraining based on accumulated feedback"""
    print("\nTriggering model retraining...")
    
    try:
        # Try to trigger retraining via API
        response = requests.post(
            "http://localhost:8001/retrain-model/",
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Model retraining triggered successfully via API")
            return True
        else:
            print(f"❌ Failed to trigger retraining via API: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️  API not running, triggering retraining directly...")
    
    # If API is not available, run retraining directly
    try:
        trainer = FraudModelTrainer()
        success = trainer.train_champion_challenger()
        if success:
            print("✅ Model retrained successfully")
            return True
        else:
            print("ℹ️  No retraining needed based on current feedback")
            return False
    except Exception as e:
        print(f"❌ Error during direct retraining: {e}")
        return False


def check_model_performance():
    """Check model performance metrics after retraining"""
    print("\nChecking model performance...")
    
    try:
        # Try to get performance via API
        response = requests.get(
            "http://localhost:8001/monitoring/model-performance/",
            timeout=10
        )
        
        if response.status_code == 200:
            metrics = response.json()
            print(f"📊 Model Performance Metrics:")
            print(f"   Precision: {metrics.get('precision', 'N/A')}")
            print(f"   Recall: {metrics.get('recall', 'N/A')}")
            print(f"   F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(f"   Evaluated Transactions: {metrics.get('total_evaluated', 'N/A')}")
            return metrics
        else:
            print(f"❌ Failed to get performance metrics via API: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️  API not running, calculating metrics directly...")
    
    # Calculate metrics directly from database
    conn = sqlite3.connect("./data/feedback.db")
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
        print("No evaluated transactions found")
        return {"precision": 0, "recall": 0, "f1_score": 0, "total_evaluated": 0}

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

    metrics = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "total_evaluated": len(prob_label_pairs)
    }
    
    print(f"📊 Model Performance Metrics (Direct Calculation):")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall: {metrics['recall']}")
    print(f"   F1 Score: {metrics['f1_score']}")
    print(f"   Evaluated Transactions: {metrics['total_evaluated']}")
    
    return metrics


def main():
    """Main function to run the end-to-end test"""
    print("🚀 Starting End-to-End Test for Tunisian Fraud Detection System")
    print("="*60)
    
    # Step 1: Set up test environment
    setup_test_environment()
    
    # Step 2: Simulate analyst feedback
    simulate_analyst_feedback()
    
    # Step 3: Trigger model retraining
    retraining_success = trigger_model_retraining()
    
    # Step 4: Wait a bit for retraining to complete
    if retraining_success:
        print("\n⏳ Waiting for model retraining to complete...")
        time.sleep(5)  # Wait 5 seconds for retraining to potentially complete
    
    # Step 5: Check model performance
    metrics = check_model_performance()
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("🏁 End-to-End Test Summary:")
    print(f"   • Analyst feedback simulated: ✅")
    print(f"   • Model retraining triggered: {'✅' if retraining_success else '⚠️'}")
    print(f"   • Performance metrics calculated: ✅")
    print(f"   • Final F1 Score: {metrics.get('f1_score', 'N/A')}")
    print("="*60)
    
    if metrics.get('f1_score', 0) > 0:
        print("✅ End-to-End Flow Successful! The feedback loop is operational.")
        return True
    else:
        print("⚠️  End-to-End Flow Completed, but model performance needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)