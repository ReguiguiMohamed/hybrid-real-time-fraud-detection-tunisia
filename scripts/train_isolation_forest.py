"""Train the complementary Isolation Forest anomaly detector from recorded alerts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ml.isolation_forest import IsolationForestAnomalyDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detector")
    parser.add_argument("--db-path", default=str(PROJECT_ROOT / "data" / "feedback.db"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "isolation_forest.joblib"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()

    data = IsolationForestAnomalyDetector.load_training_data_from_db(args.db_path, limit=args.limit)
    if len(data) < 2:
        raise SystemExit("Need at least 2 recorded alerts to train Isolation Forest")

    detector = IsolationForestAnomalyDetector(contamination=args.contamination).train(data)
    output_path = detector.save(args.output)
    print(f"Saved Isolation Forest model to {output_path}")
    print(detector.metadata_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
