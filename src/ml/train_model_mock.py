# src/ml/train_model_mock.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, rand
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import random

class FraudModelTrainerMock:
    def __init__(self):
        self.spark = SparkSession.builder.appName("TunisianFraud-ModelTrainer-Mock").getOrCreate()

    def create_mock_data(self):
        # Create mock data similar to what would come from the silver layer
        data = []
        for i in range(10000):
            v_count = random.randint(1, 10)
            g_dist = random.randint(1, 3)
            avg_amount = random.uniform(100, 10000)
            is_smurfing = 1 if 1400 <= avg_amount <= 1500 else 0
            high_velocity_flag = 1 if v_count > 5 else 0
            
            # Create a label based on fraud rules
            if v_count > 5 or g_dist > 1 or (1400 <= avg_amount <= 1500 and v_count > 3):
                label = 1  # fraud
            else:
                label = 0  # not fraud
            
            data.append((v_count, g_dist, avg_amount, is_smurfing, high_velocity_flag, label))
        
        # Create DataFrame
        df = self.spark.createDataFrame(data, ["v_count", "g_dist", "avg_amount", "is_smurfing", "high_velocity_flag", "label"])
        return df

    def train(self):
        dataset = self.create_mock_data()
        print(f"Training on dataset with {dataset.count()} records")
        
        # Define the Feature Vector (Industrial standard for Spark ML)
        feature_cols = ["v_count", "g_dist", "avg_amount", "is_smurfing", "high_velocity_flag"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
        
        # XGBoost Spark Estimator (Standard since late 2024 for distributed Tabular AI)
        xgb = SparkXGBClassifier(
            featuresCol="features",
            labelCol="label",
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1
        )
        
        pipeline = Pipeline(stages=[assembler, xgb])
        model = pipeline.fit(dataset)
        
        # Persistent 'Brain' - Saved for Day 5 Iteration 2
        model.write().overwrite().save("models/fraud_xgb_v1")
        print("✅ Gold Model (XGBoost) trained and saved to /models/fraud_xgb_v1")

if __name__ == "__main__":
    trainer = FraudModelTrainerMock()
    trainer.train()