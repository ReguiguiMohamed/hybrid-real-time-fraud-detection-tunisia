#!/usr/bin/env python3
"""
Cloud Infrastructure Cost Estimation for Amastan Fraud Shield Guard
Calculates monthly cost projections for production deployment.
Supports AWS, GCP, and Azure pricing models.

Usage:
    python scripts/cost_estimate.py
    python scripts/cost_estimate.py --cloud aws --region us-east-1 --tx-per-day 1000000
"""
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CostBreakdown:
    compute_monthly: float = 0.0
    storage_monthly: float = 0.0
    network_monthly: float = 0.0
    ml_inference_monthly: float = 0.0
    monitoring_monthly: float = 0.0
    total_monthly: float = 0.0
    annual_run_rate: float = 0.0
    cost_per_transaction: float = 0.0
    cost_per_fraud_alert: float = 0.0


def estimate_aws(region: str = "us-east-1", tx_per_day: int = 1_000_000, fraud_rate: float = 0.0001) -> CostBreakdown:
    """
    AWS cost estimation based on K8s manifest resource requests.
    Prices as of March 2026.
    """
    costs = CostBreakdown()

    # Compute: EKS + EC2 instances
    # 1x Kafka broker: m5.xlarge (4 vCPU, 16GB) ~ $0.192/hr
    # 1x Zookeeper: t3.medium (2 vCPU, 4GB) ~ $0.0416/hr
    # 1x Spark Consumer: r5.xlarge (4 vCPU, 32GB) ~ $0.252/hr
    # 2x API (min): t3.medium (2 vCPU, 4GB) ~ $0.0416/hr each
    # 1x Ollama (GPU): g5.xlarge (4 vCPU, 16GB, 1x A10G) ~ $1.006/hr
    # 1x ChromaDB: t3.small (2 vCPU, 2GB) ~ $0.0208/hr

    compute_hours = 730  # hours per month
    kafka_hourly = 0.192
    zookeeper_hourly = 0.0416
    spark_hourly = 0.252
    api_hourly = 0.0416 * 2  # 2 replicas
    ollama_hourly = 1.006  # GPU instance
    chroma_hourly = 0.0208

    costs.compute_monthly = (kafka_hourly + zookeeper_hourly + spark_hourly +
                             api_hourly + ollama_hourly + chroma_hourly) * compute_hours

    # EKS control plane
    costs.compute_monthly += 0.10 * compute_hours  # $0.10/hr for EKS

    # Storage: EBS gp3 volumes
    # Kafka: 50GB, Checkpoint: 10GB, Parquet: 50GB, Feedback: 20GB, Models: 10GB, Ollama: 30GB, Chroma: 10GB
    storage_gb = 50 + 10 + 50 + 20 + 10 + 30 + 10
    ebs_monthly_per_gb = 0.08  # gp3 per GB-month
    costs.storage_monthly = storage_gb * ebs_monthly_per_gb

    # Network: Data transfer (egress)
    # Assume 1KB per transaction, 10% of traffic goes to API responses
    daily_data_gb = (tx_per_day * 1024) / (1024 ** 3)
    monthly_data_gb = daily_data_gb * 30
    egress_per_gb = 0.09  # First 10TB/month
    costs.network_monthly = monthly_data_gb * egress_per_gb

    # ML Inference: Ollama GPU is included in compute
    # If using Bedrock instead: ~$0.002 per 1K tokens
    # Assume 500 tokens per SAR, 0.01% fraud rate
    tx_fraud_per_day = tx_per_day * fraud_rate
    sar_tokens = 500
    bedrock_per_1k = 0.002
    costs.ml_inference_monthly = (tx_fraud_per_day * 30 * sar_tokens / 1000) * bedrock_per_1k

    # Monitoring: CloudWatch + Managed Prometheus
    # CloudWatch: ~$0.30/metric-month, ~50 metrics = $15
    # Managed Prometheus: $0.90/million samples, ~1M/day = $27
    costs.monitoring_monthly = 15 + 27

    costs.total_monthly = (costs.compute_monthly + costs.storage_monthly +
                          costs.network_monthly + costs.ml_inference_monthly +
                          costs.monitoring_monthly)
    costs.annual_run_rate = costs.total_monthly * 12
    costs.cost_per_transaction = costs.total_monthly / (tx_per_day * 30)
    costs.cost_per_fraud_alert = costs.total_monthly / (tx_per_day * 30 * fraud_rate) if fraud_rate > 0 else 0

    return costs


def estimate_gcp(region: str = "us-central1", tx_per_day: int = 1_000_000, fraud_rate: float = 0.0001) -> CostBreakdown:
    """GCP cost estimation."""
    costs = CostBreakdown()

    compute_hours = 730
    # GKE + Compute Engine
    kafka_hourly = 0.170  # n2-standard-4
    zookeeper_hourly = 0.034  # e2-medium
    spark_hourly = 0.227  # n2-highmem-4
    api_hourly = 0.034 * 2  # e2-medium x2
    ollama_hourly = 1.030  # g2-standard-4 (L4 GPU)
    chroma_hourly = 0.017  # e2-small

    costs.compute_monthly = (kafka_hourly + zookeeper_hourly + spark_hourly +
                             api_hourly + ollama_hourly + chroma_hourly) * compute_hours
    costs.compute_monthly += 0.10 * compute_hours  # GKE control plane

    storage_gb = 180
    costs.storage_monthly = storage_gb * 0.08  # pd-standard

    daily_data_gb = (tx_per_day * 1024) / (1024 ** 3)
    monthly_data_gb = daily_data_gb * 30
    costs.network_monthly = monthly_data_gb * 0.08  # GCP egress

    tx_fraud_per_day = tx_per_day * fraud_rate
    costs.ml_inference_monthly = (tx_fraud_per_day * 30 * 500 / 1000) * 0.002  # Vertex AI

    costs.monitoring_monthly = 20  # Cloud Monitoring

    costs.total_monthly = (costs.compute_monthly + costs.storage_monthly +
                          costs.network_monthly + costs.ml_inference_monthly +
                          costs.monitoring_monthly)
    costs.annual_run_rate = costs.total_monthly * 12
    costs.cost_per_transaction = costs.total_monthly / (tx_per_day * 30)
    costs.cost_per_fraud_alert = costs.total_monthly / (tx_per_day * 30 * fraud_rate) if fraud_rate > 0 else 0

    return costs


def estimate_azure(region: str = "eastus", tx_per_day: int = 1_000_000, fraud_rate: float = 0.0001) -> CostBreakdown:
    """Azure cost estimation."""
    costs = CostBreakdown()

    compute_hours = 730
    # AKS + VMs
    kafka_hourly = 0.190  # Standard_D4s_v5
    zookeeper_hourly = 0.042  # Standard_B2s
    spark_hourly = 0.256  # Standard_E4s_v5
    api_hourly = 0.042 * 2
    ollama_hourly = 1.120  # Standard_NC8as_T4_v3
    chroma_hourly = 0.021  # Standard_B1s

    costs.compute_monthly = (kafka_hourly + zookeeper_hourly + spark_hourly +
                             api_hourly + ollama_hourly + chroma_hourly) * compute_hours
    costs.compute_monthly += 0.10 * compute_hours  # AKS

    storage_gb = 180
    costs.storage_monthly = storage_gb * 0.084  # Premium SSD LRS

    daily_data_gb = (tx_per_day * 1024) / (1024 ** 3)
    monthly_data_gb = daily_data_gb * 30
    costs.network_monthly = monthly_data_gb * 0.087

    tx_fraud_per_day = tx_per_day * fraud_rate
    costs.ml_inference_monthly = (tx_fraud_per_day * 30 * 500 / 1000) * 0.002  # Azure OpenAI

    costs.monitoring_monthly = 20  # Azure Monitor

    costs.total_monthly = (costs.compute_monthly + costs.storage_monthly +
                          costs.network_monthly + costs.ml_inference_monthly +
                          costs.monitoring_monthly)
    costs.annual_run_rate = costs.total_monthly * 12
    costs.cost_per_transaction = costs.total_monthly / (tx_per_day * 30)
    costs.cost_per_fraud_alert = costs.total_monthly / (tx_per_day * 30 * fraud_rate) if fraud_rate > 0 else 0

    return costs


def print_cost_report(cloud_name: str, costs: CostBreakdown, tx_per_day: int):
    """Print a formatted cost report."""
    print("\n" + "=" * 70)
    print(f"  AMASTAN FRAUD SHIELD GUARD - COST ESTIMATION ({cloud_name.upper()})")
    print("=" * 70)
    print(f"\n  Throughput: {tx_per_day:,} transactions/day")
    print(f"  Estimated fraud alerts: {int(tx_per_day * 0.0001):,}/day (0.01% fraud rate)")
    print("\n" + "-" * 70)
    print("  MONTHLY COST BREAKDOWN")
    print("-" * 70)
    print(f"  {'Compute (EKS/GKE/AKS + VMs)':.<40} ${costs.compute_monthly:>10,.2f}")
    print(f"  {'Storage (EBS/GCP Disk/Azure SSD)':.<40} ${costs.storage_monthly:>10,.2f}")
    print(f"  {'Network (Egress)':.<40} ${costs.network_monthly:>10,.2f}")
    print(f"  {'ML Inference (GPU/Bedrock)':.<40} ${costs.ml_inference_monthly:>10,.2f}")
    print(f"  {'Monitoring (Prometheus/Grafana)':.<40} ${costs.monitoring_monthly:>10,.2f}")
    print("-" * 70)
    print(f"  {'TOTAL MONTHLY':.<40} ${costs.total_monthly:>10,.2f}")
    print(f"  {'ANNUAL RUN RATE':.<40} ${costs.annual_run_rate:>10,.2f}")
    print("\n" + "-" * 70)
    print("  UNIT ECONOMICS")
    print("-" * 70)
    print(f"  {'Cost per transaction':.<40} ${costs.cost_per_transaction:>10,.6f}")
    print(f"  {'Cost per fraud alert':.<40} ${costs.cost_per_fraud_alert:>10,.4f}")
    print(f"  {'Cost per 1K transactions':.<40} ${costs.cost_per_transaction * 1000:>10,.4f}")
    print("=" * 70)
    print("\n  NOTES:")
    print("  - Prices are estimates based on on-demand rates (March 2026)")
    print("  - Savings Plans / Reserved Instances can reduce compute by 30-60%")
    print("  - GPU instance (Ollama) is the single largest cost driver")
    print("  - Consider CPU-only Ollama for cost optimization if latency allows")
    print("  - Storage assumes 180GB total with 30-day retention")
    print("  - Network costs assume 1KB per transaction")
    print("  - Monitoring costs are approximate")
    print()


def main():
    parser = argparse.ArgumentParser(description="Cloud infrastructure cost estimation")
    parser.add_argument("--cloud", choices=["aws", "gcp", "azure", "all"], default="all",
                       help="Cloud provider to estimate")
    parser.add_argument("--region", default="us-east-1", help="Cloud region")
    parser.add_argument("--tx-per-day", type=int, default=1_000_000,
                       help="Expected daily transaction volume")
    parser.add_argument("--fraud-rate", type=float, default=0.0001,
                       help="Expected fraud rate (default: 0.01%)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    estimators = {
        "aws": ("AWS (us-east-1)", estimate_aws),
        "gcp": ("GCP (us-central1)", estimate_gcp),
        "azure": ("Azure (eastus)", estimate_azure),
    }

    if args.cloud == "all":
        results = {}
        for key, (name, fn) in estimators.items():
            costs = fn(region=args.region, tx_per_day=args.tx_per_day, fraud_rate=args.fraud_rate)
            results[key] = asdict(costs)
            print_cost_report(name, costs, args.tx_per_day)

        if args.json:
            print(json.dumps(results, indent=2))
    else:
        name, fn = estimators[args.cloud]
        costs = fn(region=args.region, tx_per_day=args.tx_per_day, fraud_rate=args.fraud_rate)

        if args.json:
            print(json.dumps(asdict(costs), indent=2))
        else:
            print_cost_report(name, costs, args.tx_per_day)


if __name__ == "__main__":
    main()
