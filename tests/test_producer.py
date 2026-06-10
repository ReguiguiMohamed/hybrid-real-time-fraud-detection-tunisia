"""Tests for transaction producer."""

import json

import pytest

from producer.producer import generate_tx


class TestProducer:
    def test_generate_tx_returns_dict(self):
        tx = generate_tx()
        assert isinstance(tx, dict)

    def test_generate_tx_has_required_fields(self):
        tx = generate_tx()
        required = {
            "transaction_id",
            "timestamp",
            "user_id",
            "amount_tnd",
            "governorate",
            "payment_method",
            "branch_id",
            "fraud_seed",
        }
        assert required.issubset(tx.keys())

    def test_generate_tx_amount_range(self):
        for _ in range(100):
            tx = generate_tx()
            assert 10.0 <= tx["amount_tnd"] <= 15000.0

    def test_generate_tx_valid_governorate(self):
        valid = {"Tunis", "Sfax", "Sousse", "Ariana", "Bizerte", "Gabes", "Kairouan"}
        for _ in range(50):
            tx = generate_tx()
            assert tx["governorate"] in valid

    def test_generate_tx_valid_payment_method(self):
        valid = {"eDinar", "Flouci", "Konnect", "Carte Bancaire", "Cash on Delivery"}
        for _ in range(50):
            tx = generate_tx()
            assert tx["payment_method"] in valid

    def test_generate_tx_json_serializable(self):
        tx = generate_tx()
        serialized = json.dumps(tx)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized["transaction_id"] == tx["transaction_id"]

    def test_fraud_seed_logic(self):
        """High-value transactions (>10000) should always be fraud-seeded."""
        fraud_high_value = 0
        total_high_value = 0
        for _ in range(500):
            tx = generate_tx()
            if tx["amount_tnd"] > 10000:
                total_high_value += 1
                if tx["fraud_seed"]:
                    fraud_high_value += 1
        if total_high_value > 0:
            assert fraud_high_value == total_high_value
