# src/streaming/consumer_demo.py
# Alternative demo consumer that shows the concept without requiring full Spark on Windows
import json
import sys
from pathlib import Path

# Ensure 'src' is in path for schema import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from kafka import KafkaConsumer
from shared.schemas import Transaction

def demo_consumer():
    """
    Demo consumer that shows the concept of consuming from Kafka
    This is a simplified version that doesn't require Spark/Hadoop on Windows
    """
    print("Starting Tunisian Transaction Consumer Demo...")
    print("Connecting to Kafka broker at localhost:9092")
    
    try:
        # Create Kafka consumer
        consumer = KafkaConsumer(
            'tunisian_transactions',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest messages
            enable_auto_commit=True
        )
        
        print("Connected! Listening for transactions...")
        print("Press Ctrl+C to stop\n")
        
        # Listen for messages
        for message in consumer:
            try:
                # Parse the transaction
                tx_data = message.value
                
                # Validate against Pydantic schema (simulating what Spark would do)
                transaction = Transaction(**tx_data)
                
                # Print formatted output (similar to what Spark would show)
                print(f"RECEIVED TRANSACTION:")
                print(f"  ID: {transaction.transaction_id}")
                print(f"  Timestamp: {transaction.timestamp}")
                print(f"  User: {transaction.user_id}")
                print(f"  Amount (TND): {transaction.amount_tnd}")
                print(f"  Governorate: {transaction.governorate}")
                print(f"  Payment Method: {transaction.payment_method}")
                print(f"  Fraud Seed: {transaction.fraud_seed}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        print("Make sure:")
        print("1. Kafka is running (docker-compose up)")
        print("2. Topic 'tunisian_transactions' exists")
        print("3. Producer is sending data")

if __name__ == "__main__":
    demo_consumer()