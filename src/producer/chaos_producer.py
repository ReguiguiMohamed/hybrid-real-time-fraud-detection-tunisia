from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone, timedelta
import uuid
import argparse

from faker import Faker

from shared.schemas import Transaction

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Install with 'pip install kafka-python' for Kafka publishing.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TunisiaChaosProducer")

try:
    fake = Faker("fr_TN")
except AttributeError:
    fake = Faker("fr_FR")

GOVERNORATES = ["Tunis", "Sfax", "Sousse", "Ariana", "Bizerte", "Gabes", "Kairouan"]
METHODS = ["eDinar", "Flouci", "Konnect", "Carte Bancaire", "Cash on Delivery"]


def generate_tx() -> dict:
    amount = round(random.uniform(10, 15000), 2)
    # Fraud Seeding: High-value anomalies common in 2026 e-commerce
    is_fraud = amount > 10000 or random.random() < 0.05

    tx = Transaction(
        user_id=f"USER_{fake.random_int(min=1000, max=9999)}",
        amount_tnd=amount,
        governorate=random.choice(GOVERNORATES),
        payment_method=random.choice(METHODS),
        fraud_seed=is_fraud,
    )
    return tx.model_dump()


def generate_delayed_tx():
    """Generate a transaction with a timestamp from 12 minutes ago to test watermarking"""
    tx = generate_tx()
    # Simulate a 12-minute delay (beyond your 10-min watermark)
    past_time = datetime.now(timezone.utc) - timedelta(minutes=12)
    tx["timestamp"] = past_time.isoformat()
    return tx


def publish_to_kafka(bootstrap_servers: str = "localhost:9092", include_delayed=False):
    if not KAFKA_AVAILABLE:
        logger.error("Kafka library not available. Please install kafka-python.")
        return

    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,
            linger_ms=5,  # Small delay to allow batching
        )

        logger.info(f"Starting Tunisian Chaos Transaction Stream with Kafka publishing to {bootstrap_servers}")

        while True:
            # Randomly decide whether to send a delayed transaction
            if include_delayed and random.random() < 0.1:  # 10% chance of delayed transaction
                tx = generate_delayed_tx()
                logger.info(f"Sending delayed transaction: {tx['transaction_id']}")
            else:
                tx = generate_tx()

            # Publish to Kafka
            future = producer.send('tunisian_transactions', tx)
            try:
                # Block for 'synchronous' sends
                record_metadata = future.get(timeout=10)
                logger.info(f"Sent transaction {tx['transaction_id']} to topic {record_metadata.topic}, partition {record_metadata.partition}, offset {record_metadata.offset}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

            # Adjust sleep based on desired rate
            time.sleep(random.uniform(0.1, 0.5))  # Faster for testing

    except KeyboardInterrupt:
        logger.info("Stopping chaos producer...")
    except Exception as e:
        logger.error(f"Error in Kafka chaos producer: {e}")
    finally:
        if 'producer' in locals():
            producer.close()


def simulate_locally(rate: float = 1.0, include_delayed=False):
    logger.info("Starting Tunisian Chaos Transaction Stream (Simulation Mode)")
    interval = 1.0 / rate if rate > 0 else 1.0

    try:
        while True:
            if include_delayed and random.random() < 0.1:  # 10% chance of delayed transaction
                tx = generate_delayed_tx()
                logger.info(f"Generated delayed transaction: {tx['transaction_id']}")
            else:
                tx = generate_tx()
            
            print(json.dumps(tx, indent=2))
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopping chaos producer...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tunisian Chaos Transaction Producer")
    parser.add_argument("--rate", type=float, default=1.0, help="Transactions per second (default: 1.0)")
    parser.add_argument("--kafka", action="store_true", help="Publish to Kafka instead of printing locally")
    parser.add_argument("--include-delayed", action="store_true", help="Include delayed transactions to test watermarking")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                       help="Kafka bootstrap servers (default: localhost:9092)")

    args = parser.parse_args()

    if args.kafka:
        publish_to_kafka(args.bootstrap_servers, args.include_delayed)
    else:
        simulate_locally(args.rate, args.include_delayed)