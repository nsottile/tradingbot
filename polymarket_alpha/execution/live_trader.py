"""
LiveTrader: places real orders on Polymarket via authenticated CLOB API.

THIS MODULE IS DISABLED BY DEFAULT.
It is only activated when `paper_trading = False` in config AND
POLYMARKET_API_KEY / SECRET / PASSPHRASE are set.

ALWAYS run in paper trading mode first. Only switch to live trading
after extensive paper trading validation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx

from polymarket_alpha.config import get_config
from polymarket_alpha.execution.paper_trader import PaperTrader
from polymarket_alpha.schemas import Action, MarketSnapshot, TradeDecision
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

_LIVE_TRADING_WARNING = """
=================================================================
WARNING: LIVE TRADING IS ENABLED.
Real money will be placed on Polymarket.
You are responsible for all trades.
There are NO guarantees of profitability.
Past backtest or paper trading results do NOT predict live results.
=================================================================
"""


class LiveTrader(PaperTrader):
    """
    Live trader extending PaperTrader with real order placement.

    Inherits all paper trading logic. Only the order placement
    method is overridden to make real API calls.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config()

        if cfg.paper_trading:
            raise RuntimeError(
                "LiveTrader instantiated but paper_trading=True in config. "
                "Set PAPER_TRADING=false to enable live trading."
            )

        required = [cfg.polymarket_api_key, cfg.polymarket_api_secret, cfg.polymarket_api_passphrase]
        if not all(required):
            raise RuntimeError(
                "Live trading requires POLYMARKET_API_KEY, "
                "POLYMARKET_API_SECRET, and POLYMARKET_API_PASSPHRASE to be set."
            )

        logger.warning(_LIVE_TRADING_WARNING)

        self._api_key = cfg.polymarket_api_key
        self._api_secret = cfg.polymarket_api_secret
        self._api_passphrase = cfg.polymarket_api_passphrase
        self._clob_url = cfg.polymarket_clob_url.rstrip("/")
        self._http = httpx.Client(timeout=15.0)

    def _record_paper_trade(
        self, snap: MarketSnapshot, decision: TradeDecision
    ) -> None:
        """Override: place a real order instead of (or in addition to) recording."""
        # Always record for audit trail
        super()._record_paper_trade(snap, decision)

        # Place real order
        order_result = self._place_order(snap, decision)
        if order_result:
            logger.info(
                "Live order placed",
                extra={
                    "market_id": snap.market_id,
                    "order_id": order_result.get("orderID"),
                    "action": decision.action.value,
                    "size": decision.position_size,
                },
            )
        else:
            logger.error(
                "Live order FAILED — position NOT entered",
                extra={"market_id": snap.market_id},
            )

    def _place_order(
        self, snap: MarketSnapshot, decision: TradeDecision
    ) -> Optional[Dict]:
        """
        Place a market order via the Polymarket CLOB API.

        Returns the API response dict on success, None on failure.
        """
        if decision.action == Action.BUY_YES:
            token_id = snap.yes_token_id
            side = "BUY"
            price = snap.yes_price
        elif decision.action == Action.BUY_NO:
            token_id = snap.no_token_id
            side = "BUY"
            price = snap.no_price
        else:
            return None

        if not token_id:
            logger.error(
                "No token ID for order",
                extra={"market_id": snap.market_id, "action": decision.action.value},
            )
            return None

        # Calculate share quantity: size_usd / price_per_share
        shares = decision.position_size / max(price, 0.01)
        shares = round(shares, 2)

        order_payload = {
            "orderType": "LIMIT",
            "tokenID": token_id,
            "price": round(price + self._cfg.slippage_estimate, 4),
            "side": side,
            "size": shares,
            "expiration": 0,  # GTC
        }

        try:
            # NOTE: Real CLOB authentication requires HMAC signing.
            # This is a placeholder — implement proper auth before going live.
            # See: https://docs.polymarket.com/#authentication
            headers = {
                "POLY-API-KEY": self._api_key,
                "POLY-PASSPHRASE": self._api_passphrase,
                "Content-Type": "application/json",
            }
            resp = self._http.post(
                f"{self._clob_url}/order",
                json=order_payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error(
                "Order placement exception",
                extra={"error": str(exc), "market_id": snap.market_id},
            )
            return None

    def cancel_all_orders(self) -> None:
        """Cancel all open orders (emergency stop)."""
        try:
            headers = {
                "POLY-API-KEY": self._api_key,
                "POLY-PASSPHRASE": self._api_passphrase,
            }
            resp = self._http.delete(f"{self._clob_url}/orders", headers=headers)
            resp.raise_for_status()
            logger.info("All orders cancelled")
        except Exception as exc:
            logger.error("Failed to cancel orders", extra={"error": str(exc)})
