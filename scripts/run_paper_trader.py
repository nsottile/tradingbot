"""
scripts/run_paper_trader.py

Runs the paper trading loop.
No real money is ever placed.

Usage:
    python scripts/run_paper_trader.py
"""

import sys

sys.path.insert(0, ".")

from polymarket_alpha.execution.paper_trader import PaperTrader
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("paper_trader")


def main() -> None:
    logger.info("Paper trading loop starting")
    trader = PaperTrader()
    trader.run()  # Runs until KeyboardInterrupt


if __name__ == "__main__":
    main()
