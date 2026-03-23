from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from polymarket_alpha.data.schemas import NormalizedMarketSnapshot


class FeaturePipeline(ABC):
    @abstractmethod
    def build(self, snap: NormalizedMarketSnapshot, history: List[NormalizedMarketSnapshot]) -> Dict[str, float]:
        raise NotImplementedError


class SignalEngine(ABC):
    @abstractmethod
    def score(self, snap: NormalizedMarketSnapshot, features: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError


class DecisionPolicy(ABC):
    @abstractmethod
    def decide(self, snap: NormalizedMarketSnapshot, signals: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

