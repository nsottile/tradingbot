from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Dict, List

from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.strategies.base import FeaturePipeline


def _ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    out = values[0]
    for val in values[1:]:
        out = alpha * val + (1.0 - alpha) * out
    return out


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return sqrt(max(var, 0.0))


class DefaultFeaturePipeline(FeaturePipeline):
    """Computes technical and microstructure features for strategy scoring."""

    def build(self, snap: NormalizedMarketSnapshot, history: List[NormalizedMarketSnapshot]) -> Dict[str, float]:
        prices = [h.price for h in history[-100:]] + [snap.price]
        vols = [h.volume_24h for h in history[-50:]] + [snap.volume_24h]

        sma_10 = mean(prices[-10:]) if prices else snap.price
        sma_20 = mean(prices[-20:]) if prices else snap.price
        ema_12 = _ema(prices[-26:], 12)
        ema_26 = _ema(prices[-26:], 26)
        macd = ema_12 - ema_26

        gains = []
        losses = []
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))
        avg_gain = mean(gains[-14:]) if gains else 0.0
        avg_loss = mean(losses[-14:]) if losses else 0.0
        rs = avg_gain / max(avg_loss, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        std_20 = _std(prices[-20:])
        bb_upper = sma_20 + (2.0 * std_20)
        bb_lower = sma_20 - (2.0 * std_20)
        momentum_5 = snap.price - prices[-6] if len(prices) >= 6 else 0.0
        volume_ratio = snap.volume_24h / max(mean(vols[-20:]) if vols else 1.0, 1e-6)

        return {
            "rsi": rsi,
            "macd": macd,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "momentum_5": momentum_5,
            "volume_ratio": volume_ratio,
            "volatility": std_20,
            "spread": snap.spread,
            "liquidity": snap.liquidity,
            "sentiment": snap.sentiment_score or 0.0,
        }

