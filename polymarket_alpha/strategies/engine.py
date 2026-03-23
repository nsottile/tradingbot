from __future__ import annotations

from typing import Dict, List

from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.strategies.base import DecisionPolicy, FeaturePipeline, SignalEngine
from polymarket_alpha.strategies.features import DefaultFeaturePipeline
from polymarket_alpha.strategies.policy import DefaultDecisionPolicy
from polymarket_alpha.strategies.signals import HybridSignalEngine


class StrategyEngine:
    """Modular strategy runtime with explicit RL-upgrade boundaries."""

    def __init__(
        self,
        features: FeaturePipeline | None = None,
        signals: SignalEngine | None = None,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self._features = features or DefaultFeaturePipeline()
        self._signals = signals or HybridSignalEngine()
        self._policy = policy or DefaultDecisionPolicy()

    def evaluate(
        self,
        snap: NormalizedMarketSnapshot,
        history: List[NormalizedMarketSnapshot],
    ) -> Dict[str, Dict[str, float]]:
        feature_values = self._features.build(snap, history)
        signal_values = self._signals.score(snap, feature_values)
        policy_values = self._policy.decide(snap, signal_values)
        return {
            "features": feature_values,
            "signals": signal_values,
            "policy": policy_values,
        }

