from __future__ import annotations

from typing import Dict

from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.strategies.base import SignalEngine


class HybridSignalEngine(SignalEngine):
    """Rule-based + lightweight ML-style blend scoring."""

    def score(self, snap: NormalizedMarketSnapshot, features: Dict[str, float]) -> Dict[str, float]:
        rsi = features.get("rsi", 50.0)
        macd = features.get("macd", 0.0)
        momentum = features.get("momentum_5", 0.0)
        volume_ratio = features.get("volume_ratio", 1.0)
        volatility = features.get("volatility", 0.0)
        sentiment = features.get("sentiment", 0.0)

        # Rule component
        rule_score = 0.0
        if rsi < 35:
            rule_score += 0.2
        if rsi > 70:
            rule_score -= 0.2
        rule_score += 0.3 if macd > 0 else -0.3
        rule_score += 0.2 if momentum > 0 else -0.2

        # Lightweight linear proxy for ML confidence
        ml_raw = (
            0.7 * macd
            + 0.3 * momentum
            + 0.15 * (volume_ratio - 1.0)
            - 0.2 * volatility
            + 0.25 * sentiment
            - 0.1 * snap.spread
        )
        ml_score = max(-1.0, min(1.0, ml_raw))
        combined = (0.55 * rule_score) + (0.45 * ml_score)
        confidence = max(0.05, min(0.99, 0.55 + abs(combined) * 0.4))
        probability = max(0.01, min(0.99, 0.5 + (combined * 0.35)))

        return {
            "rule_score": rule_score,
            "ml_score": ml_score,
            "combined_score": combined,
            "probability": probability,
            "confidence": confidence,
        }

