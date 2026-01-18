from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

from faker import Faker

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.schemas import Transaction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TunisiaProducer")

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


if __name__ == "__main__":
    logger.info("Starting Tunisian Transaction Stream (Simulation Mode)")
    try:
        while True:
            tx = generate_tx()
            print(json.dumps(tx, indent=2))
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        logger.info("Stopping producer...")
