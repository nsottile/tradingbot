"""Run either legacy or autonomous paper-trading loop."""

import argparse
import sys

sys.path.insert(0, ".")

from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine
from polymarket_alpha.execution.paper_trader import PaperTrader
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("paper_trader")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["legacy", "autonomous"], default="autonomous")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    logger.info("Paper trading loop starting", extra={"mode": args.mode})
    if args.mode == "legacy":
        trader = PaperTrader()
        trader.run(max_iterations=args.iterations)
    else:
        engine = AutonomousTradingEngine()
        engine.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
