from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.schemas import CalibrationMetrics, TradeDecision


class TradingRepository:
    """Repository facade over legacy store for new engine modules."""

    def __init__(self) -> None:
        self._store = MarketStore()

    def save_decision(self, decision: TradeDecision) -> None:
        self._store.save_decision(decision)

    def get_decisions(
        self,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        return self._store.get_decisions(since=since, limit=limit)

    def save_calibration(self, metrics: CalibrationMetrics) -> None:
        self._store.save_calibration_metrics(metrics)

    def get_calibration(self, market_family: str = "global", limit: int = 200) -> List[Dict[str, Any]]:
        return self._store.get_calibration_history(market_family=market_family, limit=limit)

