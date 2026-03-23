"""
scripts/run_collector.py

Runs the market data collector on a schedule.
Fetches live Polymarket markets every `poll_interval_seconds` and
stores snapshots in PostgreSQL.

Usage:
    python scripts/run_collector.py
"""

import signal
import sys
import time

sys.path.insert(0, ".")

from polymarket_alpha.config import get_config
from polymarket_alpha.ingestion.collector import MarketCollector
from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("collector")
_running = True


def _shutdown(signum, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def main() -> None:
    cfg = get_config()
    collector = MarketCollector()
    store = MarketStore()

    logger.info("Collector started", extra={"interval": cfg.poll_interval_seconds})

    while _running:
        try:
            snapshots = collector.fetch_active_markets()
            for snap in snapshots:
                store.save_snapshot(snap)
            logger.info("Snapshots stored", extra={"count": len(snapshots)})
        except Exception as exc:
            logger.error("Collector error", extra={"error": str(exc)})

        if _running:
            time.sleep(cfg.poll_interval_seconds)

    collector.close()
    logger.info("Collector stopped")


if __name__ == "__main__":
    main()
