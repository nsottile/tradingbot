from __future__ import annotations

from typing import Dict

from polymarket_alpha.config import get_config
from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.strategies.base import DecisionPolicy


class DefaultDecisionPolicy(DecisionPolicy):
    """Policy layer designed to be replaceable by RL policy later."""

    def __init__(self) -> None:
        self._cfg = get_config()

    def decide(self, snap: NormalizedMarketSnapshot, signals: Dict[str, float]) -> Dict[str, float]:
        probability = signals.get("probability", 0.5)
        confidence = signals.get("confidence", 0.5)
        edge = abs(probability - 0.5) * 2.0
        direction = "BUY" if probability >= 0.5 else "SELL"

        should_trade = (
            confidence >= self._cfg.min_confidence
            and edge >= self._cfg.edge_threshold
            and snap.liquidity >= self._cfg.min_liquidity
            and snap.spread <= self._cfg.max_spread
        )
        return {
            "should_trade": 1.0 if should_trade else 0.0,
            "probability": probability,
            "confidence": confidence,
            "edge": edge,
            "direction": 1.0 if direction == "BUY" else -1.0,
        }

