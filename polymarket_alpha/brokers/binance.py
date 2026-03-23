"""
Binance (spot) broker stub — requires BINANCE_API_KEY / BINANCE_API_SECRET for real calls.

Uses public endpoints only when unauthenticated; signed order placement is a placeholder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import httpx

from polymarket_alpha.brokers.base import BrokerClient
from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult, OrderSide
from polymarket_alpha.config import get_config
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BinanceBrokerStub(BrokerClient):
    venue = BrokerVenue.BINANCE

    def __init__(self) -> None:
        cfg = get_config()
        self._base = (cfg.binance_rest_url or "https://api.binance.com").rstrip("/")
        self._key = cfg.binance_api_key
        self._secret = cfg.binance_api_secret
        self._http = httpx.Client(timeout=20.0)

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not self._key or not self._secret:
            return OrderResult(
                success=False,
                message="Binance keys not set — stub only (implement HMAC signing for production)",
            )
        # Production requires signed POST /api/v3/order — not implemented here by design
        logger.warning("Binance live order not implemented — use exchange UI or extend with HMAC")
        return OrderResult(success=False, message="Binance signed orders not implemented in stub")

    def cancel_order(self, order_id: str) -> bool:
        return False

    def cancel_all_orders(self) -> bool:
        return False

    def list_open_orders(self) -> List[OpenOrder]:
        return []

    def get_balances(self) -> List[BalanceSnapshot]:
        return [
            BalanceSnapshot(
                venue=BrokerVenue.BINANCE,
                currency="USDT",
                available=0.0,
                as_of=datetime.now(timezone.utc),
            )
        ]
