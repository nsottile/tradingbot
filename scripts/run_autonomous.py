"""
Run the new autonomous AI paper-trading loop.

Usage:
    python scripts/run_autonomous.py --iterations 10
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("autonomous_runner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    engine = AutonomousTradingEngine()
    logger.info("Autonomous runner starting", extra={"iterations": args.iterations})
    engine.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()

