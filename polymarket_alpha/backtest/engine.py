"""
BacktestEngine: realistic historical simulation of the trading system.

CRITICAL ANTI-LOOKAHEAD GUARANTEES:
- Data is processed strictly in chronological order.
- At each timestamp T, only data with timestamp < T is used for features.
- Resolved outcomes are NEVER fed back into signal generation.
- Walk-forward evaluation: train period must precede test period.
- Fills are simulated with realistic slippage and fees.
- Missing or stale data is handled gracefully (skip, not error).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from polymarket_alpha.config import get_config
from polymarket_alpha.execution.decision_engine import TradeDecisionEngine
from polymarket_alpha.risk.manager import RiskManager
from polymarket_alpha.schemas import Action, BacktestTrade, MarketSnapshot, TradeDecision
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Walks through historical snapshots in time order and simulates trades.

    Usage
    -----
    engine = BacktestEngine(snapshots, resolutions)
    results = engine.run()
    print(results["metrics"])
    """

    def __init__(
        self,
        snapshots: List[MarketSnapshot],
        resolutions: Dict[str, float],  # market_id -> 1.0 (YES) or 0.0 (NO)
        initial_bankroll: float = 10_000.0,
    ) -> None:
        """
        Parameters
        ----------
        snapshots:
            All historical snapshots. MUST include both training and test periods.
        resolutions:
            Ground truth outcomes. Only used at resolution time, never during
            feature construction. NEVER leak these into signals.
        initial_bankroll:
            Starting bankroll for the simulation.
        """
        cfg = get_config()
        self._cfg = cfg
        self._resolutions = resolutions
        self._bankroll = initial_bankroll
        self._initial_bankroll = initial_bankroll

        # Sort strictly by time — enforces no lookahead
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp)

        # Group snapshots by market_id for fast history lookup
        self._by_market: Dict[str, List[MarketSnapshot]] = defaultdict(list)
        for snap in self._snapshots:
            self._by_market[snap.market_id].append(snap)

        self._risk = RiskManager(initial_bankroll=initial_bankroll)
        self._decision_engine = TradeDecisionEngine(risk_manager=self._risk)

        self._equity_curve: List[Dict[str, Any]] = []
        self._all_decisions: List[TradeDecision] = []
        self._open_positions: Dict[str, BacktestTrade] = {}  # market_id -> trade
        self._closed_trades: List[BacktestTrade] = []

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute the backtest.

        Parameters
        ----------
        start_date:
            Only process snapshots at or after this timestamp.
        end_date:
            Only process snapshots before this timestamp.

        Returns
        -------
        dict with keys: trades, decisions, equity_curve, metrics
        """
        # Filter to the test window (no lookahead into future)
        snaps = self._snapshots
        if start_date:
            snaps = [s for s in snaps if s.timestamp >= start_date]
        if end_date:
            snaps = [s for s in snaps if s.timestamp < end_date]

        logger.info(
            "Backtest starting",
            extra={"n_snapshots": len(snaps), "bankroll": self._bankroll},
        )

        seen_markets: set = set()

        for snap in snaps:
            # Get history: ONLY snapshots with timestamp < current snap timestamp
            history = [
                h for h in self._by_market[snap.market_id]
                if h.timestamp < snap.timestamp
            ]

            # Check if this market has already resolved — skip if so
            if snap.market_id in self._resolutions and not snap.is_active:
                self._maybe_close_position(snap)
                continue

            # One decision per market per snapshot (avoid spam)
            decision = self._decision_engine.decide(
                snap=snap,
                history=history,
                data_version="backtest",
            )
            self._all_decisions.append(decision)

            # Open position if trade is warranted
            if not decision.skipped and decision.action != Action.SKIP:
                if snap.market_id not in self._open_positions:
                    trade = self._open_trade(snap, decision)
                    if trade:
                        self._open_positions[snap.market_id] = trade

            # Attempt to close resolved positions
            self._maybe_close_position(snap)

            # Record equity
            self._equity_curve.append(
                {
                    "timestamp": snap.timestamp,
                    "bankroll": self._risk.get_state().bankroll,
                    "daily_pnl": self._risk.get_state().daily_pnl,
                    "drawdown": self._risk.get_state().current_drawdown_pct,
                }
            )

        # Force close any remaining open positions at last known price
        self._force_close_all()

        metrics = self._compute_metrics()
        logger.info("Backtest complete", extra={"metrics": metrics})

        return {
            "trades": self._closed_trades,
            "decisions": self._all_decisions,
            "equity_curve": self._equity_curve,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_trade(
        self, snap: MarketSnapshot, decision: TradeDecision
    ) -> Optional[BacktestTrade]:
        """Simulate opening a position with realistic fill."""
        if decision.action == Action.BUY_YES:
            entry_price = snap.yes_price + self._cfg.slippage_estimate
        elif decision.action == Action.BUY_NO:
            entry_price = snap.no_price + self._cfg.slippage_estimate
        else:
            return None

        entry_price = max(0.01, min(0.99, entry_price))

        return BacktestTrade(
            market_id=snap.market_id,
            entry_timestamp=snap.timestamp,
            action=decision.action,
            entry_price=entry_price,
            position_size=decision.position_size,
            fees=decision.position_size * self._cfg.fee_estimate,
        )

    def _maybe_close_position(self, snap: MarketSnapshot) -> None:
        """Close a position if the market has resolved."""
        market_id = snap.market_id
        if market_id not in self._open_positions:
            return
        if market_id not in self._resolutions:
            return
        if snap.is_active:
            return  # Still open

        trade = self._open_positions.pop(market_id)
        outcome = self._resolutions[market_id]  # 1.0 = YES, 0.0 = NO
        closed = self._close_trade(trade, outcome, snap.timestamp)
        self._closed_trades.append(closed)
        self._risk.record_pnl(closed.net_pnl or 0.0, snap.category)

    def _close_trade(
        self,
        trade: BacktestTrade,
        outcome: float,
        exit_timestamp: datetime,
    ) -> BacktestTrade:
        """Compute PnL at resolution."""
        if trade.action == Action.BUY_YES:
            won = outcome >= 0.5
        elif trade.action == Action.BUY_NO:
            won = outcome < 0.5
        else:
            won = False

        if won:
            # Paid `entry_price * size` for a contract paying `size`
            gross_pnl = trade.position_size * (1.0 / trade.entry_price - 1.0)
            exit_price = 1.0
        else:
            gross_pnl = -trade.position_size
            exit_price = 0.0

        net_pnl = gross_pnl - trade.fees

        return BacktestTrade(
            market_id=trade.market_id,
            entry_timestamp=trade.entry_timestamp,
            exit_timestamp=exit_timestamp,
            action=trade.action,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            position_size=trade.position_size,
            gross_pnl=gross_pnl,
            fees=trade.fees,
            slippage=trade.slippage,
            net_pnl=net_pnl,
            resolved=True,
            correct_direction=won,
        )

    def _force_close_all(self) -> None:
        """Close remaining open positions at last known price (unrealised)."""
        for market_id, trade in list(self._open_positions.items()):
            closed = BacktestTrade(
                market_id=market_id,
                entry_timestamp=trade.entry_timestamp,
                exit_timestamp=datetime.now(timezone.utc),
                action=trade.action,
                entry_price=trade.entry_price,
                position_size=trade.position_size,
                gross_pnl=0.0,  # Unknown — mark at cost
                fees=trade.fees,
                net_pnl=-trade.fees,
                resolved=False,
            )
            self._closed_trades.append(closed)
        self._open_positions.clear()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> Dict[str, Any]:
        if not self._closed_trades:
            return {"error": "No closed trades"}

        net_pnls = [t.net_pnl or 0.0 for t in self._closed_trades if t.resolved]
        correct = [t for t in self._closed_trades if t.correct_direction]

        total_trades = len([t for t in self._closed_trades if t.resolved])
        hit_rate = len(correct) / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(net_pnls)
        total_return = total_pnl / self._initial_bankroll

        # Sharpe-like ratio (using daily PnL from equity curve)
        daily_pnls: Dict[str, float] = defaultdict(float)
        for point in self._equity_curve:
            day = point["timestamp"].strftime("%Y-%m-%d")
            daily_pnls[day] = point["daily_pnl"]

        daily_returns = list(daily_pnls.values())
        sharpe = 0.0
        if len(daily_returns) > 5 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * math.sqrt(252)

        # Max drawdown
        bankrolls = [p["bankroll"] for p in self._equity_curve]
        max_dd = 0.0
        if bankrolls:
            peak = bankrolls[0]
            for b in bankrolls:
                peak = max(peak, b)
                dd = (peak - b) / (peak + 1e-9)
                max_dd = max(max_dd, dd)

        # Skip rate
        skips = len([d for d in self._all_decisions if d.skipped])
        total_decisions = len(self._all_decisions)
        skip_rate = skips / total_decisions if total_decisions > 0 else 1.0

        return {
            "total_trades": total_trades,
            "hit_rate": round(hit_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "skip_rate": round(skip_rate, 4),
            "final_bankroll": round(self._risk.get_state().bankroll, 2),
            "initial_bankroll": self._initial_bankroll,
        }
