"""
Sports betting adapter stub — jurisdiction-specific; no default API.

Implement per operator (e.g. licensed APIs) when available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from polymarket_alpha.brokers.base import BrokerClient
from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult


class SportsBrokerStub(BrokerClient):
    venue = BrokerVenue.SPORTS

    def place_order(self, req: OrderRequest) -> OrderResult:
        return OrderResult(success=False, message="Sports venue not configured — legal/API integration required")

    def cancel_order(self, order_id: str) -> bool:
        return False

    def cancel_all_orders(self) -> bool:
        return False

    def list_open_orders(self) -> List[OpenOrder]:
        return []

    def get_balances(self) -> List[BalanceSnapshot]:
        return [
            BalanceSnapshot(
                venue=BrokerVenue.SPORTS,
                currency="USD",
                available=0.0,
                as_of=datetime.now(timezone.utc),
            )
        ]
