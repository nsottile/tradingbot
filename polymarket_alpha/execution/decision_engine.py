"""
TradeDecisionEngine: orchestrates all signals into a final trade decision.

Every decision is fully logged with raw inputs, features, signal probabilities,
calibrated estimate, risk state, and the final action taken.
This makes every decision reproducible from stored inputs + code version.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from polymarket_alpha.calibration.calibrator import Calibrator
from polymarket_alpha.config import get_config
from polymarket_alpha.features.builder import FeatureBuilder
from polymarket_alpha.models.ensemble import EnsembleModel
from polymarket_alpha.models.llm_model import LLMProbabilityModel
from polymarket_alpha.models.signal_models import HeuristicModel, MicrostructureModel
from polymarket_alpha.risk.manager import RiskManager
from polymarket_alpha.schemas import (
    Action,
    LLMSignal,
    MarketSnapshot,
    TradeDecision,
)
from polymarket_alpha.utils.logging_utils import get_logger, log_decision

logger = get_logger(__name__)

_MODEL_VERSION = "0.1.0"


def _compute_ev(
    calibrated_prob: float,
    action: Action,
    snap: MarketSnapshot,
    slippage: float,
    fee: float,
) -> float:
    """
    Expected value of a trade after fees and slippage.

    EV = P(win) * net_payout - P(lose) * stake
    net_payout = (1/price - 1) * stake  → but stake=1 for normalisation

    Returns EV as a fraction of stake.
    """
    if action == Action.BUY_YES:
        price = snap.yes_price + slippage
        p = calibrated_prob
    elif action == Action.BUY_NO:
        price = snap.no_price + slippage
        p = 1.0 - calibrated_prob
    else:
        return 0.0

    price = max(0.01, min(0.99, price))
    net_odds = (1.0 / price) - 1.0

    ev = p * net_odds - (1.0 - p)
    ev -= fee  # Subtract trading fee
    return ev


class TradeDecisionEngine:
    """
    Full decision pipeline: features → signals → ensemble → calibration → risk → decision.
    """

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        llm_model: Optional[LLMProbabilityModel] = None,
    ) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._feature_builder = FeatureBuilder()
        self._llm_model = llm_model or LLMProbabilityModel()
        self._micro_model = MicrostructureModel()
        self._heuristic_model = HeuristicModel()
        self._ensemble = EnsembleModel()
        self._calibrator = Calibrator()
        self._risk = risk_manager or RiskManager()

    def decide(
        self,
        snap: MarketSnapshot,
        history: Optional[List[MarketSnapshot]] = None,
        data_version: str = "live",
    ) -> TradeDecision:
        """
        Run the full decision pipeline for a single market snapshot.

        Parameters
        ----------
        snap:
            Current market state.
        history:
            Prior snapshots for feature calculation (oldest first, no lookahead).
        data_version:
            Identifier for the data batch (for reproducibility).

        Returns
        -------
        TradeDecision with action=SKIP if no trade is warranted.
        """
        # 1. Build features
        features = self._feature_builder.build(snap, history)

        # 2. Get individual signals
        llm_signal: Optional[LLMSignal] = None
        llm_prob: Optional[float] = None
        llm_conf: Optional[float] = None

        if self._cfg.llm_enabled:
            llm_signal = self._llm_model.predict(snap, features)
            if llm_signal:
                llm_prob = llm_signal.probability
                llm_conf = llm_signal.confidence

        micro_prob, micro_conf = self._micro_model.predict(snap, features)
        heuristic_prob, heuristic_conf = self._heuristic_model.predict(snap, features)

        # 3. Ensemble
        ensemble_prob, ensemble_conf, signal_probs = self._ensemble.predict(
            snap=snap,
            llm_prob=llm_prob,
            llm_conf=llm_conf,
            micro_prob=micro_prob,
            micro_conf=micro_conf,
            heuristic_prob=heuristic_prob,
            heuristic_conf=heuristic_conf,
        )

        # 4. Calibrate
        calibrated_prob = self._calibrator.calibrate(
            ensemble_prob, market_family=snap.category
        )

        # 5. Determine best action and compute EV
        ev_yes = _compute_ev(
            calibrated_prob, Action.BUY_YES, snap,
            self._cfg.slippage_estimate, self._cfg.fee_estimate,
        )
        ev_no = _compute_ev(
            calibrated_prob, Action.BUY_NO, snap,
            self._cfg.slippage_estimate, self._cfg.fee_estimate,
        )

        if ev_yes >= ev_no and ev_yes > 0:
            best_action = Action.BUY_YES
            best_ev = ev_yes
        elif ev_no > ev_yes and ev_no > 0:
            best_action = Action.BUY_NO
            best_ev = ev_no
        else:
            # No positive EV — skip
            decision = self._make_skip_decision(
                snap=snap,
                signal_probs=signal_probs,
                calibrated_prob=calibrated_prob,
                ev=max(ev_yes, ev_no),
                confidence=ensemble_conf,
                features=features,
                reason="No positive expected value after costs",
                data_version=data_version,
            )
            return decision

        # 6. Risk check and sizing
        allowed, position_size, risk_reason = self._risk.check_and_size(
            snap=snap,
            action=best_action,
            calibrated_prob=calibrated_prob,
            edge_after_costs=best_ev,
            confidence=ensemble_conf,
        )

        if not allowed:
            decision = self._make_skip_decision(
                snap=snap,
                signal_probs=signal_probs,
                calibrated_prob=calibrated_prob,
                ev=best_ev,
                confidence=ensemble_conf,
                features=features,
                reason=f"Risk check failed: {risk_reason}",
                data_version=data_version,
            )
            return decision

        # 7. Execute trade (record in risk manager)
        self._risk.record_trade(snap.category, position_size)

        decision = TradeDecision(
            market_id=snap.market_id,
            timestamp=datetime.now(timezone.utc),
            signal_probs=signal_probs,
            calibrated_prob=calibrated_prob,
            expected_value=best_ev,
            edge_after_costs=best_ev,
            confidence=ensemble_conf,
            position_size=position_size,
            action=best_action,
            reason=risk_reason,
            model_version=_MODEL_VERSION,
            data_version=data_version,
            skipped=False,
        )

        log_decision(
            logger=logger,
            market_id=snap.market_id,
            action=best_action.value,
            reason=risk_reason,
            features={k: round(v, 4) for k, v in features.items()},
            signal_probs={
                "llm": llm_prob,
                "micro": micro_prob,
                "heuristic": heuristic_prob,
                "ensemble": ensemble_prob,
                "calibrated": calibrated_prob,
            },
            risk_state=self._risk.get_state().model_dump(),
        )

        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_skip_decision(
        self,
        snap: MarketSnapshot,
        signal_probs: Any,
        calibrated_prob: float,
        ev: float,
        confidence: float,
        features: Dict[str, float],
        reason: str,
        data_version: str,
    ) -> TradeDecision:
        decision = TradeDecision(
            market_id=snap.market_id,
            timestamp=datetime.now(timezone.utc),
            signal_probs=signal_probs,
            calibrated_prob=calibrated_prob,
            expected_value=ev,
            edge_after_costs=ev,
            confidence=confidence,
            position_size=0.0,
            action=Action.SKIP,
            reason=reason,
            model_version=_MODEL_VERSION,
            data_version=data_version,
            skipped=True,
            skip_reason=reason,
        )
        log_decision(
            logger=logger,
            market_id=snap.market_id,
            action=Action.SKIP.value,
            reason=reason,
            features={k: round(v, 4) for k, v in features.items()},
            signal_probs=signal_probs.model_dump() if hasattr(signal_probs, "model_dump") else {},
        )
        return decision
