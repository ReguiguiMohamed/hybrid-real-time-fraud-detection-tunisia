#!/usr/bin/env python3
"""
Bootstrap script to initialize the fraud detection system with a base model
and populate the model registry for immediate explainability availability.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json
from pathlib import Path

def create_mock_transactions(n_samples=500):
    """Generate mock transactions for initial model training"""
    np.random.seed(42)  # For reproducible results
    
    transactions = []
    for i in range(n_samples):
        # Simulate features that might indicate fraud
        v_count = np.random.poisson(5)  # Velocity count
        g_dist = np.random.uniform(0, 1000)  # Geographic distance
        avg_amount = np.random.lognormal(8, 1)  # Average transaction amount
        is_smurfing = np.random.choice([0, 1], p=[0.95, 0.05])  # Structuring pattern
        high_velocity_flag = np.random.choice([0, 1], p=[0.9, 0.1])  # High velocity flag
        
        # Create a synthetic fraud label based on feature combinations
        risk_score = (
            (v_count / 10) * 0.3 +
            (g_dist / 1000) * 0.2 +
            (min(avg_amount, 5000) / 5000) * 0.3 +
            is_smurfing * 0.1 +
            high_velocity_flag * 0.1
        )
        
        fraud_label = 1 if risk_score > 0.5 else 0
        
        transaction = {
            'transaction_id': f'TXN_{i:06d}',
            'user_id': f'USER_{np.random.randint(1000, 9999)}',
            'amount_tnd': round(avg_amount, 2),
            'governorate': np.random.choice(['Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte']),
            'payment_method': np.random.choice(['mobile', 'card', 'bank_transfer']),
            'branch_id': f'B{i % 10:02d}',
            'timestamp': (datetime.now() - timedelta(minutes=np.random.randint(1, 1440))).isoformat(),
            'v_count': v_count,
            'g_dist': g_dist,
            'avg_amount': avg_amount,
            'is_smurfing': is_smurfing,
            'high_velocity_flag': high_velocity_flag,
            'risk_score': risk_score,
            'is_fraud': fraud_label
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def train_base_model(df):
    """Train a base model on the mock data"""
    # Prepare features
    feature_cols = ['v_count', 'g_dist', 'avg_amount', 'is_smurfing', 'high_velocity_flag']
    X = df[feature_cols].values
    y = df['is_fraud'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    # Get feature importances
    feature_names = feature_cols
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    
    return model, accuracy, feature_importance_dict

def save_model_and_register(model, accuracy, feature_importances):
    """Save the model and register it in the database"""
    # Create models directory if it doesn't exist
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / 'v1_champion_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Connect to database
    db_path = Path('./data/feedback.db')
    db_path.parent.mkdir(exist_ok=True)  # Create data directory if needed
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create model_registry table if it doesn't exist
    cursor.execute('''
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
    ''')
    
    # Insert the champion model
    cursor.execute('''
        INSERT OR REPLACE INTO model_registry 
        (version_id, model_path, f1_score, auc, is_champion, promoted_at, training_samples_count, feature_importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        'v1_champion',
        str(model_path),
        accuracy,
        0.85,  # Placeholder AUC
        1,  # Is champion
        datetime.now().isoformat(),
        500,  # Training samples
        json.dumps(feature_importances)
    ))
    
    conn.commit()
    conn.close()
    
    print(f"Model saved to {model_path}")
    print(f"Model registered in database with feature importances: {feature_importances}")

def main():
    print("Bootstrapping fraud detection system...")
    
    # Generate mock transactions
    print("Generating mock transactions...")
    df = create_mock_transactions(500)
    print(f"Generated {len(df)} mock transactions")
    
    # Train base model
    print("Training base model...")
    model, accuracy, feature_importances = train_base_model(df)
    print(f"Base model trained with accuracy: {accuracy:.3f}")
    
    # Save model and register
    print("Saving model and registering...")
    save_model_and_register(model, accuracy, feature_importances)
    
    print("System bootstrapped successfully!")
    print("Champion model is now available for immediate use.")

if __name__ == "__main__":
    main()