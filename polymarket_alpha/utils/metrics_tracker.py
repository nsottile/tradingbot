"""
MetricsTracker: computes and stores rolling performance metrics.
Runs as a periodic job separate from the trading loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from polymarket_alpha.calibration.calibrator import Calibrator
from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Computes rolling calibration metrics from stored decisions
    and resolved market outcomes.

    Call `update()` periodically (e.g., daily) after market resolution data
    has been ingested.
    """

    def __init__(self) -> None:
        self._store = MarketStore()
        self._calibrator = Calibrator()

    def update(
        self,
        resolved_outcomes: Dict[str, float],  # market_id -> 1.0/0.0
        market_family: str = "global",
    ) -> Optional[Dict[str, Any]]:
        """
        Match stored decisions against resolved outcomes and compute metrics.

        Parameters
        ----------
        resolved_outcomes:
            Ground truth outcomes keyed by market_id.
        market_family:
            Which family to compute metrics for.

        Returns
        -------
        dict of metrics, or None if insufficient data.
        """
        decisions = self._store.get_decisions(limit=10_000)
        if not decisions:
            return None

        probs: List[float] = []
        outcomes: List[float] = []

        for d in decisions:
            mid = d.get("market_id")
            if mid in resolved_outcomes:
                p = d.get("calibrated_prob")
                if p is not None:
                    probs.append(float(p))
                    outcomes.append(resolved_outcomes[mid])

        if len(probs) < 5:
            logger.info(
                "Insufficient resolved outcomes for metrics",
                extra={"n": len(probs)},
            )
            return None

        metrics = self._calibrator.evaluate(probs, outcomes, market_family=market_family)
        if metrics:
            self._store.save_calibration_metrics(metrics)
            logger.info(
                "Calibration metrics updated",
                extra=metrics.model_dump(),
            )
            return metrics.model_dump()

        return None

    def compute_hit_rate(
        self,
        resolved_outcomes: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute directional accuracy for executed trades."""
        decisions = self._store.get_decisions(limit=10_000)
        correct = 0
        total = 0

        for d in decisions:
            if d.get("skipped"):
                continue
            mid = d.get("market_id")
            action = d.get("action", "")
            outcome = resolved_outcomes.get(mid)
            if outcome is None:
                continue

            if action == "BUY_YES" and outcome >= 0.5:
                correct += 1
                total += 1
            elif action == "BUY_NO" and outcome < 0.5:
                correct += 1
                total += 1
            elif action in ("BUY_YES", "BUY_NO"):
                total += 1

        return {
            "hit_rate": round(correct / total, 4) if total > 0 else 0.0,
            "n_resolved": total,
        }
