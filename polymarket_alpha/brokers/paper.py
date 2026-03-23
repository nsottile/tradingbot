"""Paper execution backed by PortfolioLedger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from polymarket_alpha.brokers.base import BrokerClient
from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult, OrderSide
from polymarket_alpha.engine.ledger import PortfolioLedger


class PaperBroker(BrokerClient):
    venue = BrokerVenue.PAPER

    def __init__(self, ledger: PortfolioLedger) -> None:
        self._ledger = ledger

    def place_order(self, req: OrderRequest) -> OrderResult:
        if req.notional_usd <= 0 or req.limit_price <= 0:
            return OrderResult(success=False, message="Invalid notional or price")
        if req.side == OrderSide.BUY:
            self._ledger.execute_buy(req.market_id, req.symbol, req.notional_usd, req.limit_price)
            return OrderResult(success=True, order_id=f"paper-{req.market_id}", message="Paper fill")
        # SELL: close position
        self._ledger.close_position(req.market_id)
        return OrderResult(success=True, order_id=f"paper-close-{req.market_id}", message="Paper close")

    def cancel_order(self, order_id: str) -> bool:
        return True

    def cancel_all_orders(self) -> bool:
        return True

    def list_open_orders(self) -> List[OpenOrder]:
        return []

    def get_balances(self) -> List[BalanceSnapshot]:
        return [
            BalanceSnapshot(
                venue=BrokerVenue.PAPER,
                currency="USD",
                available=self._ledger.cash,
                as_of=datetime.now(timezone.utc),
            )
        ]
