"""
FeatureBuilder: constructs predictive features from market snapshots.

CRITICAL: All features are derived from data that was available at the
time of the snapshot. No future information is ever used.
Features that require historical data only look backward in time.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from polymarket_alpha.schemas import MarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Seconds in common time horizons
_HOUR = 3600
_DAY = 86400


class FeatureBuilder:
    """
    Computes a feature vector for a given market snapshot,
    optionally using a history of prior snapshots for the same market.

    All features must be causally valid: they may only use information
    that was observable at `snapshot.timestamp`.
    """

    def build(
        self,
        snapshot: MarketSnapshot,
        history: Optional[List[MarketSnapshot]] = None,
    ) -> Dict[str, float]:
        """
        Build a feature dictionary for the given snapshot.

        Parameters
        ----------
        snapshot:
            The current market state.
        history:
            Previous snapshots for this market, ordered oldest-first.
            Must not contain any snapshot with timestamp > snapshot.timestamp.

        Returns
        -------
        dict of str -> float
            All values are finite floats ready for model ingestion.
        """
        if history is None:
            history = []

        # Validate no lookahead in history
        history = [h for h in history if h.timestamp < snapshot.timestamp]

        feats: Dict[str, float] = {}

        feats.update(self._price_features(snapshot))
        feats.update(self._microstructure_features(snapshot))
        feats.update(self._time_features(snapshot))
        feats.update(self._momentum_features(snapshot, history))
        feats.update(self._volatility_features(snapshot, history))
        feats.update(self._volume_features(snapshot, history))
        feats.update(self._category_features(snapshot))
        feats.update(self._heuristic_features(snapshot))

        # Replace any NaN / inf with 0
        feats = {k: _safe_float(v) for k, v in feats.items()}
        return feats

    # ------------------------------------------------------------------
    # Sub-feature groups
    # ------------------------------------------------------------------

    def _price_features(self, snap: MarketSnapshot) -> Dict[str, float]:
        p = snap.mid_price
        return {
            "mid_price": p,
            "yes_price": snap.yes_price,
            "no_price": snap.no_price,
            # Distance from the extremes – markets near 0 or 1 behave differently
            "price_entropy": -p * math.log(p + 1e-9) - (1 - p) * math.log(1 - p + 1e-9),
            "price_dist_from_0.5": abs(p - 0.5),
            "price_gt_0.9": float(p > 0.9),
            "price_lt_0.1": float(p < 0.1),
        }

    def _microstructure_features(self, snap: MarketSnapshot) -> Dict[str, float]:
        spread = snap.spread
        liq = snap.liquidity
        return {
            "spread": spread,
            "log_liquidity": math.log1p(liq),
            "spread_over_price": spread / (snap.mid_price + 1e-9),
            "liquidity_gt_1000": float(liq > 1_000),
            "liquidity_gt_10000": float(liq > 10_000),
        }

    def _time_features(self, snap: MarketSnapshot) -> Dict[str, float]:
        now = snap.timestamp
        feats: Dict[str, float] = {}

        if snap.resolution_date and snap.resolution_date > now:
            ttl = (snap.resolution_date - now).total_seconds()
            feats["ttl_seconds"] = ttl
            feats["log_ttl"] = math.log1p(ttl)
            feats["ttl_days"] = ttl / _DAY
            feats["ttl_lt_1day"] = float(ttl < _DAY)
            feats["ttl_lt_7days"] = float(ttl < 7 * _DAY)
        else:
            feats["ttl_seconds"] = 0.0
            feats["log_ttl"] = 0.0
            feats["ttl_days"] = 0.0
            feats["ttl_lt_1day"] = 0.0
            feats["ttl_lt_7days"] = 0.0

        # Time-of-day (UTC hour normalized)
        feats["hour_sin"] = math.sin(2 * math.pi * now.hour / 24)
        feats["hour_cos"] = math.cos(2 * math.pi * now.hour / 24)
        # Day-of-week
        feats["dow_sin"] = math.sin(2 * math.pi * now.weekday() / 7)
        feats["dow_cos"] = math.cos(2 * math.pi * now.weekday() / 7)
        return feats

    def _momentum_features(
        self,
        snap: MarketSnapshot,
        history: List[MarketSnapshot],
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if len(history) < 2:
            return {
                "momentum_1h": 0.0,
                "momentum_24h": 0.0,
                "momentum_7d": 0.0,
            }

        current_price = snap.mid_price
        now = snap.timestamp

        for label, seconds in [("1h", _HOUR), ("24h", _DAY), ("7d", 7 * _DAY)]:
            cutoff = now.timestamp() - seconds
            past = [h for h in history if h.timestamp.timestamp() >= cutoff]
            if past:
                feats[f"momentum_{label}"] = current_price - past[0].mid_price
            else:
                feats[f"momentum_{label}"] = 0.0

        return feats

    def _volatility_features(
        self,
        snap: MarketSnapshot,
        history: List[MarketSnapshot],
    ) -> Dict[str, float]:
        if len(history) < 5:
            return {"price_volatility_24h": 0.0, "price_volatility_7d": 0.0}

        prices = pd.Series([h.mid_price for h in history] + [snap.mid_price])
        returns = prices.diff().dropna()
        now = snap.timestamp

        feats: Dict[str, float] = {}
        for label, seconds in [("24h", _DAY), ("7d", 7 * _DAY)]:
            cutoff = now.timestamp() - seconds
            window_hist = [h for h in history if h.timestamp.timestamp() >= cutoff]
            if len(window_hist) >= 3:
                w_prices = [h.mid_price for h in window_hist]
                feats[f"price_volatility_{label}"] = float(np.std(w_prices))
            else:
                feats[f"price_volatility_{label}"] = 0.0

        return feats

    def _volume_features(
        self,
        snap: MarketSnapshot,
        history: List[MarketSnapshot],
    ) -> Dict[str, float]:
        vol = snap.volume
        feats = {
            "log_volume": math.log1p(vol),
            "volume_gt_10k": float(vol > 10_000),
            "volume_gt_100k": float(vol > 100_000),
        }
        if len(history) >= 2:
            prev_vol = history[-1].volume
            feats["volume_change"] = vol - prev_vol
            feats["volume_change_pct"] = (vol - prev_vol) / (prev_vol + 1e-9)
        else:
            feats["volume_change"] = 0.0
            feats["volume_change_pct"] = 0.0
        return feats

    def _category_features(self, snap: MarketSnapshot) -> Dict[str, float]:
        """One-hot encode known categories."""
        known = ["politics", "crypto", "sports", "economics", "science", "entertainment"]
        cat = snap.category.lower()
        return {f"cat_{c}": float(cat == c) for c in known}

    def _heuristic_features(self, snap: MarketSnapshot) -> Dict[str, float]:
        """
        Simple domain heuristics.
        These encode known patterns without using future information.
        """
        p = snap.mid_price
        return {
            # Prices near 0 or 1 are often over-confident in thin markets
            "extreme_price_with_low_liquidity": float(
                (p < 0.05 or p > 0.95) and snap.liquidity < 1_000
            ),
            # High spread relative to price suggests low conviction
            "high_relative_spread": float(snap.spread / (p + 1e-9) > 0.15),
            # No resolution date – harder to arbitrage timing
            "missing_expiry_date": float(snap.resolution_date is None),
        }


def _safe_float(v: Any) -> float:
    """Convert to float, replacing NaN/inf with 0."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0
