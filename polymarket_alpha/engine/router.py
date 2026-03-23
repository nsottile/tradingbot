"""
Routes autonomous orders to paper or live brokers.

Gates (all must allow live):
- ENABLE_LIVE_TRADING=true
- PAPER_TRADING=false in config
- control.simulation_mode=false
- control.live_trading_confirmed=true (UI acknowledgment)
- not control.kill_switch
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polymarket_alpha.brokers.paper import PaperBroker
from polymarket_alpha.brokers.polymarket_live import PolymarketClobBroker
from polymarket_alpha.brokers.types import BrokerVenue, OrderRequest, OrderResult, OrderSide
from polymarket_alpha.config import get_config
from polymarket_alpha.data.schemas import NormalizedMarketSnapshot
from polymarket_alpha.engine.ledger import PortfolioLedger
from polymarket_alpha.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from polymarket_alpha.engine.control import AIControlState

logger = get_logger(__name__)


class ExecutionRouter:
    def __init__(self, ledger: PortfolioLedger, control: AIControlState) -> None:
        self._ledger = ledger
        self.control = control
        self._paper = PaperBroker(ledger)
        self._polymarket = PolymarketClobBroker()

    def uses_paper_path(self) -> bool:
        """True if the next route would use paper (simulation) execution."""
        return self._use_paper_only()

    def _use_paper_only(self) -> bool:
        cfg = get_config()
        if self.control.kill_switch:
            return True
        if self.control.simulation_mode:
            return True
        if not cfg.enable_live_trading:
            return True
        if cfg.paper_trading:
            return True
        if not self.control.live_trading_confirmed:
            return True
        return False

    def route_buy_prediction(
        self,
        snap: NormalizedMarketSnapshot,
        notional_usd: float,
        limit_price: float,
    ) -> OrderResult:
        cfg = get_config()
        cap = min(
            max(notional_usd, 0.0),
            cfg.max_order_notional_usd,
            self._ledger.cash * 0.999,
        )
        if cap <= 0 or limit_price <= 0:
            return OrderResult(success=False, message="Invalid size or price")

        req = OrderRequest(
            venue=BrokerVenue.POLYMARKET,
            market_id=snap.market_id,
            symbol=snap.symbol,
            side=OrderSide.BUY,
            notional_usd=cap,
            limit_price=limit_price,
            extra=dict(snap.raw_payload),
        )

        if self._use_paper_only():
            logger.info(
                "ExecutionRouter: paper path",
                extra={"market_id": snap.market_id, "notional": cap},
            )
            return self._paper.place_order(req)

        logger.warning(
            "ExecutionRouter: LIVE Polymarket path",
            extra={"market_id": snap.market_id, "notional": cap},
        )
        return self._polymarket.place_order(req)

    def emergency_cancel_all_live(self) -> None:
        """Kill switch helper — cancel open orders on live venues."""
        try:
            self._polymarket.cancel_all_orders()
        except Exception as exc:
            logger.error("emergency_cancel polymarket failed", extra={"error": str(exc)})
