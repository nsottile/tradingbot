from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from polymarket_alpha.config import get_config
from polymarket_alpha.data.pipeline import UnifiedMarketDataPipeline
from polymarket_alpha.data.repository import TradingRepository
from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.engine.analytics import compute_metrics
from polymarket_alpha.engine.control import AIControlState
from polymarket_alpha.engine.ledger import PortfolioLedger
from polymarket_alpha.engine.router import ExecutionRouter
from polymarket_alpha.engine.risk import RISK_PROFILES, position_size
from polymarket_alpha.schemas import Action, SignalProbabilities, TradeDecision
from polymarket_alpha.strategies.engine import StrategyEngine
from polymarket_alpha.utils.logging_utils import get_logger, log_autonomous_event
from polymarket_alpha.utils.runtime_config import get_runtime_config

logger = get_logger(__name__)


class AutonomousTradingEngine:
    """Autonomous fetch→analyze→decide→execute→log loop; routes via ExecutionRouter (paper or live)."""

    def __init__(self) -> None:
        cfg = get_config()
        runtime_cfg = get_runtime_config()
        self._cfg = cfg
        self._data = UnifiedMarketDataPipeline()
        self._repo = TradingRepository()
        self._strategy = StrategyEngine()
        self.ledger = PortfolioLedger(initial_capital=cfg.initial_bankroll)
        self.control = AIControlState(
            risk_level=runtime_cfg.default_risk_level,
            strategy_name=runtime_cfg.default_strategy,
            interval_seconds=runtime_cfg.default_loop_interval_seconds or cfg.poll_interval_seconds,
        )
        self._router = ExecutionRouter(self.ledger, self.control)
        self.history: Dict[str, List[NormalizedMarketSnapshot]] = defaultdict(list)
        self.activity_feed: List[str] = []
        self.equity_curve: List[float] = [cfg.initial_bankroll]

    def run_cycle(self) -> Dict[str, float]:
        snapshots = self._data.fetch_live()
        for snap in snapshots:
            self.ledger.mark_price(snap.market_id, snap.price)

            hist = self.history.get(snap.market_id, [])
            evaluated = self._strategy.evaluate(snap, hist)
            policy = evaluated["policy"]
            conf = policy["confidence"]

            self.history[snap.market_id].append(snap)
            self.history[snap.market_id] = self.history[snap.market_id][-200:]

            if not self.control.autonomous_mode:
                continue

            risk_profile = RISK_PROFILES.get(self.control.risk_level, RISK_PROFILES["medium"])
            notional_capital = self.ledger.equity * self.control.capital_allocation_pct
            size = position_size(notional_capital, conf, self.control.risk_level)
            action = Action.SKIP
            reason = "Policy rejected trade"

            if policy["should_trade"] > 0 and size > 0:
                action = Action.BUY_YES
                reason = "Hybrid policy approved"
                paper_before = self._router.uses_paper_path()
                exec_res = self._router.route_buy_prediction(snap, size, snap.price)
                if exec_res.success:
                    if not paper_before:
                        # Mirror fill into ledger for dashboard PnL (exchange is source of truth for risk)
                        self.ledger.execute_buy(snap.market_id, snap.symbol, size, snap.price)
                    mode = "paper" if paper_before else "live"
                    self.activity_feed.append(
                        f"[{mode}] AI bought {snap.symbol} @ {snap.price:.4f} — {exec_res.message}"
                        f" ({datetime.now(timezone.utc).isoformat()})"
                    )
                else:
                    action = Action.SKIP
                    reason = f"Execution failed: {exec_res.message}"
                    self.activity_feed.append(f"Skip {snap.symbol}: {exec_res.message}")

            self.ledger.maybe_stop_loss(snap.market_id, risk_profile.stop_loss_pct)
            decision = TradeDecision(
                market_id=snap.market_id,
                timestamp=datetime.now(timezone.utc),
                signal_probs=SignalProbabilities(
                    llm=None,
                    microstructure=evaluated["signals"]["ml_score"],
                    heuristic=evaluated["signals"]["rule_score"],
                    ensemble=evaluated["signals"]["probability"],
                ),
                calibrated_prob=evaluated["signals"]["probability"],
                expected_value=policy["edge"],
                edge_after_costs=policy["edge"],
                confidence=conf,
                position_size=size if action != Action.SKIP else 0.0,
                action=action,
                reason=reason,
                model_version="autonomous-hybrid-v1",
                data_version="live",
                skipped=(action == Action.SKIP),
                skip_reason=reason if action == Action.SKIP else None,
            )
            self._repo.save_decision(decision)

        self.equity_curve.append(self.ledger.equity)
        metrics = compute_metrics(self.ledger, self.equity_curve)
        log_autonomous_event(logger, "autonomous_cycle_complete", metrics)
        return metrics

    def run(self, max_iterations: int | None = None) -> None:
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Autonomous engine stopped by user")
                return
            except Exception as exc:
                logger.error("Autonomous cycle failed", extra={"error": str(exc)})
            iterations += 1
            if max_iterations is None or iterations < max_iterations:
                time.sleep(self.control.interval_seconds)

