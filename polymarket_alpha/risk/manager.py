"""
RiskManager: enforces all position sizing and risk limits.

Non-negotiable rules enforced here:
- Fractional Kelly only (never full Kelly)
- Max position per market
- Max category exposure
- Max daily loss
- Max drawdown
- Liquidity-aware sizing
- Confidence-based shrinking
- "No trade" zone when edge is below threshold
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Dict, Optional, Tuple

from polymarket_alpha.config import get_config
from polymarket_alpha.schemas import Action, MarketSnapshot, RiskState
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Central risk management.

    Call `check_and_size()` before every trade.
    The method returns (allowed, position_size, reason).
    """

    def __init__(self, initial_bankroll: Optional[float] = None) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._bankroll = initial_bankroll or cfg.initial_bankroll
        self._peak_bankroll = self._bankroll
        self._daily_pnl: float = 0.0
        self._last_reset_date: date = datetime.now(timezone.utc).date()
        self._category_exposure: Dict[str, float] = {}
        self._halted = False
        self._halt_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def check_and_size(
        self,
        snap: MarketSnapshot,
        action: Action,
        calibrated_prob: float,
        edge_after_costs: float,
        confidence: float,
    ) -> Tuple[bool, float, str]:
        """
        Determine if a trade is allowed and what size to use.

        Returns
        -------
        (allowed, position_size_usd, reason)
        """
        self._maybe_reset_daily()

        # 1. System halt
        if self._halted:
            return False, 0.0, f"System halted: {self._halt_reason}"

        # 2. Edge threshold
        if edge_after_costs < self._cfg.edge_threshold:
            return False, 0.0, (
                f"Edge {edge_after_costs:.4f} below threshold {self._cfg.edge_threshold}"
            )

        # 3. Confidence threshold
        if confidence < self._cfg.min_confidence:
            return False, 0.0, (
                f"Confidence {confidence:.3f} below minimum {self._cfg.min_confidence}"
            )

        # 4. Liquidity check
        if snap.liquidity < self._cfg.min_liquidity:
            return False, 0.0, (
                f"Liquidity {snap.liquidity:.0f} below minimum {self._cfg.min_liquidity}"
            )

        # 5. Spread check
        if snap.spread > self._cfg.max_spread:
            return False, 0.0, (
                f"Spread {snap.spread:.4f} above maximum {self._cfg.max_spread}"
            )

        # 6. Category exposure
        cat = snap.category
        cat_exp = self._category_exposure.get(cat, 0.0)
        cat_limit = self._bankroll * self._cfg.max_category_exposure_pct
        if cat_exp >= cat_limit:
            return False, 0.0, (
                f"Category '{cat}' exposure {cat_exp:.0f} at limit {cat_limit:.0f}"
            )

        # 7. Daily loss limit
        daily_loss_limit = self._bankroll * self._cfg.max_daily_loss_pct
        if self._daily_pnl <= -daily_loss_limit:
            self._halt("Daily loss limit reached")
            return False, 0.0, "Daily loss limit breached — system halted"

        # 8. Compute Kelly size
        position_size = self._kelly_size(
            calibrated_prob=calibrated_prob,
            action=action,
            confidence=confidence,
            snap=snap,
        )

        if position_size < 1.0:  # Less than $1 — not worth trading
            return False, 0.0, f"Position size {position_size:.2f} too small"

        reason = (
            f"Edge={edge_after_costs:.4f} conf={confidence:.3f} "
            f"size={position_size:.2f} action={action.value}"
        )
        return True, position_size, reason

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def _kelly_size(
        self,
        calibrated_prob: float,
        action: Action,
        confidence: float,
        snap: MarketSnapshot,
    ) -> float:
        """
        Fractional Kelly position sizing.

        Kelly formula for a binary bet:
            f* = (p * b - q) / b
        where:
            p = probability of winning
            q = 1 - p
            b = net odds (payout - 1)

        We then apply:
        - Kelly fraction (e.g., 0.25)
        - Confidence shrink
        - Max position cap
        - Category exposure cap
        - Liquidity cap
        """
        if action == Action.BUY_YES:
            p = calibrated_prob
            price = snap.yes_price
        elif action == Action.BUY_NO:
            p = 1.0 - calibrated_prob
            price = snap.no_price
        else:
            return 0.0

        q = 1.0 - p
        if price <= 0.001 or price >= 0.999:
            return 0.0

        # Payout: if price=0.30, winning $1 returns $1/0.30 - 1 = 2.33 net
        b = (1.0 / price) - 1.0

        kelly_f = (p * b - q) / b
        kelly_f = max(0.0, kelly_f)

        # Apply fractional Kelly
        kelly_f *= self._cfg.kelly_fraction

        # Shrink by confidence
        kelly_f *= confidence

        # Convert to dollar amount
        dollar_size = kelly_f * self._bankroll

        # Apply caps
        max_position = self._bankroll * self._cfg.max_position_pct
        dollar_size = min(dollar_size, max_position)

        # Cap by remaining category headroom
        cat = snap.category
        cat_used = self._category_exposure.get(cat, 0.0)
        cat_limit = self._bankroll * self._cfg.max_category_exposure_pct
        remaining_cat = max(0.0, cat_limit - cat_used)
        dollar_size = min(dollar_size, remaining_cat)

        # Cap by liquidity (don't exceed 5% of available liquidity)
        liq_cap = snap.liquidity * 0.05
        dollar_size = min(dollar_size, liq_cap)

        return max(0.0, dollar_size)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def record_trade(self, category: str, size: float) -> None:
        """Record an executed trade for exposure tracking."""
        self._category_exposure[category] = (
            self._category_exposure.get(category, 0.0) + size
        )

    def record_pnl(self, pnl: float, category: Optional[str] = None) -> None:
        """Update daily PnL and bankroll after a trade resolves."""
        self._daily_pnl += pnl
        self._bankroll += pnl
        self._peak_bankroll = max(self._peak_bankroll, self._bankroll)

        if category:
            # Reduce category exposure by the settled amount
            current = self._category_exposure.get(category, 0.0)
            self._category_exposure[category] = max(0.0, current - abs(pnl))

        # Check drawdown
        if self._peak_bankroll > 0:
            drawdown = (self._peak_bankroll - self._bankroll) / self._peak_bankroll
            if drawdown >= self._cfg.max_drawdown_pct:
                self._halt(f"Max drawdown {drawdown:.2%} exceeded")

    def get_state(self) -> RiskState:
        """Return a snapshot of the current risk state."""
        drawdown = 0.0
        if self._peak_bankroll > 0:
            drawdown = (self._peak_bankroll - self._bankroll) / self._peak_bankroll

        return RiskState(
            timestamp=datetime.now(timezone.utc),
            bankroll=self._bankroll,
            daily_pnl=self._daily_pnl,
            daily_loss_limit=self._bankroll * self._cfg.max_daily_loss_pct,
            max_drawdown_pct=self._cfg.max_drawdown_pct,
            current_drawdown_pct=drawdown,
            category_exposure=dict(self._category_exposure),
            halted=self._halted,
            halt_reason=self._halt_reason,
        )

    def resume(self) -> None:
        """Manually resume after a halt (e.g., after operator review)."""
        self._halted = False
        self._halt_reason = None
        logger.info("Risk manager resumed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _halt(self, reason: str) -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            logger.error("RISK HALT", extra={"reason": reason, "bankroll": self._bankroll})

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._last_reset_date = today
            logger.info("Daily PnL reset", extra={"date": str(today)})
