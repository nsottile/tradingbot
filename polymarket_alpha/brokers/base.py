"""Abstract broker client — all live venues implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult


class BrokerClient(ABC):
    venue: BrokerVenue

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult:
        """Submit an order. Implementations must validate size/price bounds."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by id."""

    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """Emergency: cancel all open orders for this venue."""

    @abstractmethod
    def list_open_orders(self) -> List[OpenOrder]:
        """Return open orders (stub may return [])."""

    @abstractmethod
    def get_balances(self) -> List[BalanceSnapshot]:
        """Return available balances per currency (stub may return [])."""
