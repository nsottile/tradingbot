"""
scripts/run_backtest.py

Runs a walk-forward backtest using stored historical snapshots.

Usage:
    python scripts/run_backtest.py --start 2023-01-01 --end 2024-01-01
"""

import argparse
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from polymarket_alpha.backtest.engine import BacktestEngine
from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.schemas import MarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    store = MarketStore()
    raw_snaps = store.get_snapshots(since=start_dt, limit=100_000)

    if not raw_snaps:
        logger.error("No historical snapshots found. Run the collector first.")
        sys.exit(1)

    snapshots = []
    for row in raw_snaps:
        try:
            snap = MarketSnapshot(**row)
            snapshots.append(snap)
        except Exception as exc:
            logger.warning("Skipping malformed snapshot", extra={"error": str(exc)})

    # NOTE: resolutions must be supplied separately in production.
    # In this demo we pass an empty dict — add your resolution data here.
    resolutions: dict = {}

    engine = BacktestEngine(
        snapshots=snapshots,
        resolutions=resolutions,
        initial_bankroll=args.bankroll,
    )

    results = engine.run(start_date=start_dt, end_date=end_dt)

    print("\n=== Backtest Results ===")
    print(json.dumps(results["metrics"], indent=2))

    trades_file = "backtest_trades.json"
    with open(trades_file, "w") as f:
        trade_rows = [t.model_dump() for t in results["trades"]]
        for row in trade_rows:
            # Convert datetime objects for JSON serialisation
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
        json.dump(trade_rows, f, indent=2, default=str)
    print(f"\nTrades saved to {trades_file}")


if __name__ == "__main__":
    main()
