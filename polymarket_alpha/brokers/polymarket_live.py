"""
Polymarket CLOB live broker.

Requires POLYMARKET_API_* env vars. HMAC signing may be incomplete — see Polymarket docs.
Never logs secrets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from polymarket_alpha.brokers.base import BrokerClient
from polymarket_alpha.brokers.types import BalanceSnapshot, BrokerVenue, OpenOrder, OrderRequest, OrderResult, OrderSide
from polymarket_alpha.config import get_config
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _payload_tokens(extra: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float, float]:
    yes_t = extra.get("yes_token_id") or extra.get("_yes_token_id")
    no_t = extra.get("no_token_id") or extra.get("_no_token_id")
    yp = float(extra.get("yes_price") or extra.get("_yes_price") or 0.5)
    np = float(extra.get("no_price") or extra.get("_no_price") or 0.5)
    return (
        str(yes_t) if yes_t else None,
        str(no_t) if no_t else None,
        yp,
        np,
    )


class PolymarketClobBroker(BrokerClient):
    venue = BrokerVenue.POLYMARKET

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._clob_url = cfg.polymarket_clob_url.rstrip("/")
        self._http = httpx.Client(timeout=15.0)

    def _headers(self) -> Dict[str, str]:
        return {
            "POLY-API-KEY": self._cfg.polymarket_api_key or "",
            "POLY-PASSPHRASE": self._cfg.polymarket_api_passphrase or "",
            "Content-Type": "application/json",
        }

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not all([self._cfg.polymarket_api_key, self._cfg.polymarket_api_passphrase]):
            return OrderResult(success=False, message="Missing Polymarket API credentials")

        yes_t, no_t, yes_p, no_p = _payload_tokens(req.extra)
        if req.side == OrderSide.BUY:
            token_id = yes_t
            price = req.limit_price if req.limit_price > 0 else yes_p
        else:
            token_id = no_t
            price = req.limit_price if req.limit_price > 0 else no_p

        if not token_id:
            return OrderResult(success=False, message="No token_id in order context")

        shares = req.notional_usd / max(price, 0.01)
        shares = round(shares, 2)
        order_payload = {
            "orderType": "LIMIT",
            "tokenID": token_id,
            "price": round(price + self._cfg.slippage_estimate, 4),
            "side": "BUY",
            "size": shares,
            "expiration": 0,
        }
        try:
            resp = self._http.post(
                f"{self._clob_url}/order",
                json=order_payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            oid = str(data.get("orderID") or data.get("order_id") or "")
            logger.info(
                "Polymarket order placed",
                extra={"market_id": req.market_id, "order_id": oid or "unknown"},
            )
            return OrderResult(success=True, order_id=oid or None, message="Submitted", raw=data)
        except Exception as exc:
            logger.error("Polymarket order failed", extra={"error": str(exc), "market_id": req.market_id})
            return OrderResult(success=False, message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = self._http.delete(
                f"{self._clob_url}/order/{order_id}",
                headers=self._headers(),
            )
            return resp.status_code < 400
        except Exception:
            return False

    def cancel_all_orders(self) -> bool:
        try:
            resp = self._http.delete(f"{self._clob_url}/orders", headers=self._headers())
            return resp.status_code < 400
        except Exception:
            return False

    def list_open_orders(self) -> List[OpenOrder]:
        # Stub: CLOB list endpoint varies; return empty until wired to official API
        return []

    def get_balances(self) -> List[BalanceSnapshot]:
        return [
            BalanceSnapshot(
                venue=BrokerVenue.POLYMARKET,
                currency="USDC",
                available=0.0,
                as_of=datetime.now(timezone.utc),
            )
        ]
