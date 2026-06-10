#!/usr/bin/env python3
"""
Load testing script for the fraud detection system
Tests the system's ability to handle high transaction volumes
"""

import argparse
import asyncio
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp


class FraudDetectionLoadTester:
    def __init__(self, api_url: str, api_token: str, rate: int = 10):
        """
        Initialize the load tester

        Args:
            api_url: Base URL for the fraud detection API
            api_token: Authentication token for the API
            rate: Target transactions per second
        """
        self.api_url = api_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self.rate = rate  # Transactions per second
        self.running = False
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "latencies": [],
            "start_time": None,
            "end_time": None,
        }
        self.lock = threading.Lock()

    async def send_single_transaction(self, session: aiohttp.ClientSession, transaction_data: Dict):
        """Send a single transaction to the API"""
        start_time = time.time()

        try:
            url = f"{self.api_url}/alerts/add/"
            async with session.post(url, json=transaction_data, headers=self.headers) as response:
                response_time = time.time() - start_time
                status = response.status

                with self.lock:
                    self.stats["total_requests"] += 1
                    if status == 200:
                        self.stats["successful_requests"] += 1
                    else:
                        self.stats["failed_requests"] += 1
                    self.stats["latencies"].append(response_time)

                if status != 200:
                    print(f"Failed to send transaction {transaction_data['transaction_id']}: {status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")

        except Exception as e:
            with self.lock:
                self.stats["total_requests"] += 1
                self.stats["failed_requests"] += 1
            print(f"Error sending transaction {transaction_data['transaction_id']}: {str(e)}")

    def generate_transaction(self) -> Dict:
        """Generate a realistic transaction for testing"""
        transaction_id = f"LOAD_TEST_TXN_{int(time.time() * 1000000)}"
        user_id = f"USER_{random.randint(10000, 99999)}"

        # Generate realistic transaction data
        transaction = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "amount_tnd": round(random.uniform(50, 5000), 2),  # Amount in Tunisian Dinars
            "governorate": random.choice(
                [
                    "Tunis",
                    "Sfax",
                    "Sousse",
                    "Kairouan",
                    "Bizerte",
                    "Gabès",
                    "Ariana",
                    "Manouba",
                    "Nabeul",
                    "Zaghouan",
                    "Ben Arous",
                    "Medenine",
                    "Tataouine",
                    "Gafsa",
                    "Tozeur",
                    "Kasserine",
                    "Sidi Bouzid",
                    "Siliana",
                    "Kebili",
                    "Jendouba",
                    "Beja",
                    "Le Kef",
                    "Mahdia",
                    "Monastir",
                    "Skanes",
                ]
            ),
            "payment_method": random.choice(["mobile", "card", "bank_transfer"]),
            "branch_id": f"B{random.randint(1, 50):02d}",
            "timestamp": datetime.now().isoformat(),
            "ml_probability": round(random.uniform(0.1, 0.99), 4),
            "sar_report": None,
            "alert_type": random.choice(["high_risk", "random_sample", "uncertainty_sample"]),
        }

        return transaction

    async def run_load_test(self, duration: int = 60):
        """
        Run the load test for a specified duration

        Args:
            duration: Duration of the test in seconds
        """
        print(f"Starting load test for {duration} seconds at {self.rate} TPS...")
        print(f"Target: {self.rate * duration} total transactions")

        self.running = True
        self.stats["start_time"] = time.time()

        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()

            while time.time() - start_time < duration and self.running:
                # Calculate how many transactions to send in this second
                start_second = time.time()

                # Send transactions at the target rate
                tasks = []
                for _ in range(self.rate):
                    if not self.running:
                        break
                    transaction = self.generate_transaction()
                    task = self.send_single_transaction(session, transaction)
                    tasks.append(task)

                # Execute all tasks concurrently
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Sleep to maintain the target rate
                elapsed = time.time() - start_second
                sleep_time = max(0, 1.0 - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        self.stats["end_time"] = time.time()
        self.print_results()

    def print_results(self):
        """Print the load test results"""
        duration = self.stats["end_time"] - self.stats["start_time"]

        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Target Rate: {self.rate} TPS")
        print(f"Actual Rate: {self.stats['total_requests']/duration:.2f} TPS")
        print(f"Total Requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")

        success_rate = (self.stats["successful_requests"] / max(1, self.stats["total_requests"])) * 100
        print(f"Success Rate: {success_rate:.2f}%")

        if self.stats["latencies"]:
            avg_latency = sum(self.stats["latencies"]) / len(self.stats["latencies"])
            p95_latency = (
                sorted(self.stats["latencies"])[int(0.95 * len(self.stats["latencies"]))]
                if self.stats["latencies"]
                else 0
            )
            max_latency = max(self.stats["latencies"]) if self.stats["latencies"] else 0

            print(f"Avg Latency: {avg_latency:.3f}s ({avg_latency*1000:.1f}ms)")
            print(f"P95 Latency: {p95_latency:.3f}s ({p95_latency*1000:.1f}ms)")
            print(f"Max Latency: {max_latency:.3f}s ({max_latency*1000:.1f}ms)")

        print("=" * 60)

    def stop(self):
        """Stop the load test"""
        self.running = False


async def main():
    parser = argparse.ArgumentParser(description="Load test the fraud detection system")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--api-token", default="default_dev_token", help="API authentication token")
    parser.add_argument("--rate", type=int, default=10, help="Transactions per second (default: 10)")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")

    args = parser.parse_args()

    tester = FraudDetectionLoadTester(args.api_url, args.api_token, args.rate)

    try:
        await tester.run_load_test(args.duration)
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
        tester.stop()
    except Exception as e:
        print(f"Error during load test: {str(e)}")
        tester.stop()


if __name__ == "__main__":
    asyncio.run(main())
