"""Generate deterministic backtest artifact for CI."""

import json
import pathlib
import tempfile

import pandas as pd

from scripts.backtest import BacktestEngine


def main():
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    parquet_path = tmp_dir / "silver_fraud_alerts"
    parquet_path.mkdir(parents=True)

    df = pd.DataFrame([
        {"amount_tnd": 16000.0, "payment_method": "card", "v_count": 1, "g_dist": 1, "label": 1, "timestamp": "2026-06-01"},
        {"amount_tnd": 100.0, "payment_method": "card", "v_count": 1, "g_dist": 1, "label": 0, "timestamp": "2026-06-01"},
    ])
    df.to_parquet(parquet_path / "data.parquet")

    engine = BacktestEngine(parquet_path=str(parquet_path / "data.parquet"))
    result = engine.run()
    pathlib.Path("backtest-report.json").write_text(json.dumps({
        "total_transactions": result.total_transactions,
        "label_source": result.label_source,
        "recommendation": result.recommendation,
    }, indent=2))
    print(f"Backtest artifact: {result.total_transactions} txns, label={result.label_source}")


if __name__ == "__main__":
    main()
