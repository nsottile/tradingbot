"""
Alpaca broker stub — wire to https://paper-api.alpaca.markets or live API when keys set.

Set ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL (optional).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import httpx

from polymarket_alpha.brokers.base import BrokerClient
from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult, OrderSide
from polymarket_alpha.config import get_config
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AlpacaBrokerStub(BrokerClient):
    venue = BrokerVenue.ALPACA

    def __init__(self) -> None:
        cfg = get_config()
        self._base = (cfg.alpaca_base_url or "https://paper-api.alpaca.markets").rstrip("/")
        self._key = cfg.alpaca_api_key
        self._secret = cfg.alpaca_api_secret
        self._http = httpx.Client(timeout=20.0)

    def _headers(self) -> Optional[dict]:
        if not self._key or not self._secret:
            return None
        return {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Content-Type": "application/json",
        }

    def place_order(self, req: OrderRequest) -> OrderResult:
        headers = self._headers()
        if not headers:
            return OrderResult(success=False, message="Alpaca keys not configured (stub)")
        try:
            body = {
                "symbol": req.symbol,
                "qty": str(max(req.notional_usd / max(req.limit_price, 1e-9), 1e-6)),
                "side": req.side.value.lower(),
                "type": "market",
                "time_in_force": "day",
            }
            r = self._http.post(
                f"{self._base}/v2/orders",
                json=body,
                headers=headers,
            )
            if r.status_code >= 400:
                return OrderResult(success=False, message=r.text[:200])
            data = r.json()
            return OrderResult(success=True, order_id=data.get("id"), raw=data)
        except Exception as exc:
            logger.error("Alpaca order stub error", extra={"error": str(exc)})
            return OrderResult(success=False, message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        headers = self._headers()
        if not headers:
            return False
        try:
            r = self._http.delete(f"{self._base}/v2/orders/{order_id}", headers=headers)
            return r.status_code < 400
        except Exception:
            return False

    def cancel_all_orders(self) -> bool:
        headers = self._headers()
        if not headers:
            return False
        try:
            r = self._http.delete(f"{self._base}/v2/orders", headers=headers)
            return r.status_code < 400
        except Exception:
            return False

    def list_open_orders(self) -> List[OpenOrder]:
        return []

    def get_balances(self) -> List[BalanceSnapshot]:
        headers = self._headers()
        if not headers:
            return []
        try:
            r = self._http.get(f"{self._base}/v2/account", headers=headers)
            if r.status_code >= 400:
                return []
            d = r.json()
            cash = float(d.get("cash", 0) or 0)
            return [
                BalanceSnapshot(
                    venue=BrokerVenue.ALPACA,
                    currency="USD",
                    available=cash,
                    as_of=datetime.now(timezone.utc),
                )
            ]
        except Exception:
            return []
