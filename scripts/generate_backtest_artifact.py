"""Generate deterministic backtest artifact for CI."""

import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest import BacktestEngine


def main():
    df = pd.DataFrame(
        [
            {
                "amount_tnd": 16000.0,
                "payment_method": "card",
                "v_count": 1,
                "g_dist": 1,
                "label": 1,
                "timestamp": "2026-06-01",
            },
            {
                "amount_tnd": 100.0,
                "payment_method": "card",
                "v_count": 1,
                "g_dist": 1,
                "label": 0,
                "timestamp": "2026-06-01",
            },
        ]
    )

    engine = BacktestEngine()

    def load_fixture(*_args, **_kwargs):
        return df.copy()

    engine.load_data = load_fixture
    result = engine.run()
    pathlib.Path("backtest-report.json").write_text(
        json.dumps(
            {
                "total_transactions": result.total_transactions,
                "label_source": result.label_source,
                "recommendation": result.recommendation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Backtest artifact: {result.total_transactions} txns, label={result.label_source}")


if __name__ == "__main__":
    main()
