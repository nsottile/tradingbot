"""
PaperTrader: simulates live trading without real money.

Runs the full decision pipeline against live market data,
records all decisions and simulated fills, and tracks a
paper bankroll. No real orders are ever placed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from polymarket_alpha.config import get_config
from polymarket_alpha.execution.decision_engine import TradeDecisionEngine
from polymarket_alpha.ingestion.collector import MarketCollector
from polymarket_alpha.ingestion.store import MarketStore
from polymarket_alpha.risk.manager import RiskManager
from polymarket_alpha.schemas import Action, MarketSnapshot, TradeDecision
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PaperTrader:
    """
    Live paper trading loop.

    Fetches real market data every `poll_interval` seconds,
    runs the decision engine, and records simulated trades.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._collector = MarketCollector()
        self._store = MarketStore()
        self._risk = RiskManager()
        self._engine = TradeDecisionEngine(risk_manager=self._risk)
        self._history: Dict[str, List[MarketSnapshot]] = {}
        self._paper_trades: List[Dict] = []

    def run(self, max_iterations: Optional[int] = None) -> None:
        """
        Run the paper trading loop.

        Parameters
        ----------
        max_iterations:
            Stop after this many poll cycles (None = run forever).
        """
        iteration = 0
        logger.info(
            "Paper trading started",
            extra={
                "bankroll": self._cfg.initial_bankroll,
                "poll_interval": self._cfg.poll_interval_seconds,
            },
        )

        while max_iterations is None or iteration < max_iterations:
            try:
                self._poll_and_decide()
            except KeyboardInterrupt:
                logger.info("Paper trader stopped by user")
                break
            except Exception as exc:
                logger.error("Poll error", extra={"error": str(exc)})

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                time.sleep(self._cfg.poll_interval_seconds)

        logger.info(
            "Paper trading complete",
            extra={
                "iterations": iteration,
                "risk_state": self._risk.get_state().model_dump(),
            },
        )

    def _poll_and_decide(self) -> None:
        """One poll cycle: fetch markets → decide → record."""
        snapshots = self._collector.fetch_active_markets()

        for snap in snapshots:
            # Stale data check
            if not self._collector.is_data_fresh(snap):
                logger.warning(
                    "Stale snapshot skipped",
                    extra={"market_id": snap.market_id},
                )
                continue

            # Store snapshot
            self._store.save_snapshot(snap)

            # Update rolling history (no lookahead: history has snapshots from PREVIOUS polls)
            history = self._history.get(snap.market_id, [])

            # Make decision
            decision = self._engine.decide(snap, history=history)

            # Store decision
            self._store.save_decision(decision)

            # Record paper trade
            if not decision.skipped and decision.action != Action.SKIP:
                self._record_paper_trade(snap, decision)

            # Update history for next poll (append current snap AFTER decision)
            self._history.setdefault(snap.market_id, []).append(snap)
            # Keep only last 1000 snapshots per market to limit memory
            self._history[snap.market_id] = self._history[snap.market_id][-1000:]

    def _record_paper_trade(
        self, snap: MarketSnapshot, decision: TradeDecision
    ) -> None:
        record = {
            "market_id": snap.market_id,
            "question": snap.question,
            "timestamp": decision.timestamp.isoformat(),
            "action": decision.action.value,
            "position_size": decision.position_size,
            "entry_price": (
                snap.yes_price if decision.action == Action.BUY_YES else snap.no_price
            ),
            "calibrated_prob": decision.calibrated_prob,
            "edge": decision.edge_after_costs,
            "confidence": decision.confidence,
        }
        self._paper_trades.append(record)
        logger.info("Paper trade recorded", extra=record)

    def get_summary(self) -> Dict:
        return {
            "total_trades": len(self._paper_trades),
            "risk_state": self._risk.get_state().model_dump(),
            "recent_trades": self._paper_trades[-10:],
        }
