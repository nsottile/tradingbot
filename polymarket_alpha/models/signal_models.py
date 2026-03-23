"""
MicrostructureModel and HeuristicModel: structural signal layers.

These models produce probability estimates based on:
- Order book and price microstructure (MicrostructureModel)
- Domain-specific heuristics and market family knowledge (HeuristicModel)

Neither model uses future information.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from polymarket_alpha.schemas import MarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MicrostructureModel:
    """
    Derives a probability signal from price microstructure.

    The core idea: if the market mid-price is a reasonable probability
    estimate, how should we adjust it given spread, liquidity, and
    short-term momentum?

    Returns (probability, confidence) in [0,1].
    """

    def predict(
        self,
        snap: MarketSnapshot,
        features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Return (probability_estimate, confidence).

        The base estimate is the market mid-price itself, adjusted
        for microstructure signals. Low liquidity / wide spread
        → shrink toward 0.5, reduce confidence.
        """
        p = snap.mid_price
        confidence = 1.0

        # Wide spread → low confidence
        spread_ratio = snap.spread / (p + 1e-9)
        if spread_ratio > 0.20:
            confidence *= 0.5
        elif spread_ratio > 0.10:
            confidence *= 0.75

        # Low liquidity → shrink toward 0.5 and reduce confidence
        liq = snap.liquidity
        if liq < 100:
            shrink = 0.6
            confidence *= 0.4
        elif liq < 1_000:
            shrink = 0.85
            confidence *= 0.7
        elif liq < 10_000:
            shrink = 0.95
            confidence *= 0.85
        else:
            shrink = 1.0

        p_adjusted = 0.5 + shrink * (p - 0.5)

        # Apply momentum from features if available
        if features:
            mom_24h = features.get("momentum_24h", 0.0)
            # Small momentum signal — only meaningful if liquidity is reasonable
            if liq > 1_000:
                p_adjusted = p_adjusted + 0.3 * mom_24h
                p_adjusted = max(0.01, min(0.99, p_adjusted))

        confidence = max(0.05, min(1.0, confidence))
        p_adjusted = max(0.01, min(0.99, p_adjusted))

        return p_adjusted, confidence


class HeuristicModel:
    """
    Domain-specific heuristic signals.

    These encode structural knowledge about market families
    (politics, crypto, sports, etc.) without using future information.

    Returns (probability_estimate, confidence) in [0,1].
    Confidence is kept low — these are weak priors.
    """

    # Heuristics by category: (prior_bias_toward_yes, confidence)
    _CATEGORY_PRIORS: Dict[str, Tuple[float, float]] = {
        "politics": (0.50, 0.20),     # Roughly even, hard to call
        "crypto": (0.45, 0.15),       # Slight bearish bias in uncertain markets
        "sports": (0.50, 0.25),       # Near even; home advantage captured by market
        "economics": (0.50, 0.20),
        "science": (0.55, 0.15),      # Scientific consensus tends toward YES
        "entertainment": (0.50, 0.10),
    }

    def predict(
        self,
        snap: MarketSnapshot,
        features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Return (probability_estimate, confidence)."""
        cat = snap.category.lower()
        base_p, base_conf = self._CATEGORY_PRIORS.get(cat, (0.50, 0.10))

        p = snap.mid_price

        # Very near expiry: market price is strongly informative
        if features:
            ttl_days = features.get("ttl_days", 99)
            if ttl_days < 1:
                # Near resolution: market price dominates, heuristic contributes little
                blend_w = 0.05
            elif ttl_days < 7:
                blend_w = 0.15
            else:
                blend_w = 0.25
        else:
            blend_w = 0.20

        # Blend: (1-w)*market_price + w*category_prior
        p_blended = (1 - blend_w) * p + blend_w * base_p

        # Additional adjustment: extreme prices on thin markets get pulled in
        if features:
            extreme_thin = features.get("extreme_price_with_low_liquidity", 0.0)
            if extreme_thin:
                p_blended = 0.5 + 0.7 * (p_blended - 0.5)  # shrink toward 0.5
                base_conf *= 0.5

        p_blended = max(0.01, min(0.99, p_blended))
        confidence = max(0.05, min(0.40, base_conf))  # Heuristics are weak

        return p_blended, confidence
