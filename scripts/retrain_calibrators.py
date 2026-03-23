"""
scripts/retrain_calibrators.py

Retrains calibrators from stored trade decisions matched to resolved outcomes.
Uses walk-forward splits — never trains on future data.

Usage:
    python scripts/retrain_calibrators.py --resolutions resolutions.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from polymarket_alpha.calibration.calibrator import Calibrator
from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger("retrain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolutions",
        required=True,
        help="Path to JSON file: {market_id: 1.0 or 0.0}",
    )
    args = parser.parse_args()

    with open(args.resolutions) as f:
        resolutions: dict = json.load(f)

    store = MarketStore()
    decisions = store.get_decisions(limit=100_000)

    if not decisions:
        logger.error("No decisions in store")
        sys.exit(1)

    # Group by category
    by_category: dict = {}
    all_probs, all_outcomes = [], []

    for d in decisions:
        mid = d.get("market_id")
        outcome = resolutions.get(mid)
        if outcome is None:
            continue
        prob = d.get("calibrated_prob")
        if prob is None:
            continue

        all_probs.append(float(prob))
        all_outcomes.append(float(outcome))

        cat = d.get("category", "global")
        by_category.setdefault(cat, ([], []))
        by_category[cat][0].append(float(prob))
        by_category[cat][1].append(float(outcome))

    if not all_probs:
        logger.error("No matched outcomes found")
        sys.exit(1)

    cal = Calibrator()

    # Global calibrator
    cal.fit(all_probs, all_outcomes, market_family="global")
    logger.info("Global calibrator trained", extra={"n": len(all_probs)})

    # Per-category calibrators
    for cat, (probs, outcomes) in by_category.items():
        if len(probs) >= 30:
            cal.fit(probs, outcomes, market_family=cat)
            logger.info(f"Category calibrator trained: {cat}", extra={"n": len(probs)})
        else:
            logger.info(f"Skipping {cat} — only {len(probs)} samples")

    # Evaluate
    metrics = cal.evaluate(all_probs, all_outcomes, market_family="global")
    if metrics:
        store.save_calibration_metrics(metrics)
        print(f"\nGlobal Brier Score: {metrics.brier_score:.4f}")
        print(f"Log Loss:           {metrics.log_loss:.4f}")
        print(f"ECE:                {metrics.ece:.4f}")
        print(f"Overconfidence:     {metrics.overconfidence:.4f}")


if __name__ == "__main__":
    main()
