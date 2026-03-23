"""
MarketCollector: pulls live market data from Polymarket public endpoints.

Endpoints used:
- Gamma API  → market discovery, metadata, historical prices
- CLOB API   → orderbook, spread, real-time prices

No authenticated endpoints are used for data ingestion.
Only order placement (LiveTrader) uses authenticated endpoints.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from polymarket_alpha.config import get_config
from polymarket_alpha.schemas import MarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MarketCollector:
    """
    Fetches and normalises live market data from Polymarket.

    Usage
    -----
    collector = MarketCollector()
    snapshots = collector.fetch_active_markets()
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._gamma_url = cfg.polymarket_gamma_url.rstrip("/")
        self._clob_url = cfg.polymarket_clob_url.rstrip("/")
        self._max_markets = cfg.max_markets_per_poll
        self._stale_threshold = cfg.stale_data_threshold_seconds
        self._client = httpx.Client(timeout=20.0, follow_redirects=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_active_markets(self) -> List[MarketSnapshot]:
        """Return snapshots for all active, liquid markets."""
        raw_markets = self._get_gamma_markets()
        snapshots: List[MarketSnapshot] = []

        for raw in raw_markets[: self._max_markets]:
            try:
                snap = self._normalise(raw)
                if snap is not None:
                    snapshots.append(snap)
            except Exception as exc:
                logger.warning(
                    "Failed to normalise market",
                    extra={"market_id": raw.get("id"), "error": str(exc)},
                )

        logger.info(
            "Fetched active markets",
            extra={"count": len(snapshots)},
        )
        return snapshots

    def fetch_market_by_id(self, market_id: str) -> Optional[MarketSnapshot]:
        """Fetch a single market snapshot by ID."""
        try:
            raw = self._get_gamma_market(market_id)
            return self._normalise(raw)
        except Exception as exc:
            logger.error(
                "Failed to fetch market by id",
                extra={"market_id": market_id, "error": str(exc)},
            )
            return None

    def fetch_clob_prices(self, token_id: str) -> Dict[str, float]:
        """
        Fetch best bid/ask from the CLOB for a given token.
        Returns {"best_bid": float, "best_ask": float, "mid": float, "spread": float}
        """
        try:
            resp = self._client.get(f"{self._clob_url}/book", params={"token_id": token_id})
            resp.raise_for_status()
            book = resp.json()
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid, "spread": spread}
        except Exception as exc:
            logger.warning(
                "CLOB price fetch failed",
                extra={"token_id": token_id, "error": str(exc)},
            )
            return {"best_bid": 0.0, "best_ask": 1.0, "mid": 0.5, "spread": 1.0}

    def is_data_fresh(self, snapshot: MarketSnapshot) -> bool:
        """Return True if the snapshot was taken within the stale threshold."""
        age = (datetime.now(timezone.utc) - snapshot.timestamp).total_seconds()
        return age <= self._stale_threshold

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_gamma_markets(self) -> List[Dict[str, Any]]:
        """Paginate through the Gamma markets endpoint."""
        markets: List[Dict[str, Any]] = []
        limit = 100
        offset = 0

        while len(markets) < self._max_markets:
            try:
                resp = self._client.get(
                    f"{self._gamma_url}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": limit,
                        "offset": offset,
                    },
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                markets.extend(batch)
                if len(batch) < limit:
                    break
                offset += limit
                time.sleep(0.1)  # Respectful rate limiting
            except httpx.HTTPError as exc:
                logger.error(
                    "Gamma API error",
                    extra={"error": str(exc), "offset": offset},
                )
                break

        return markets

    def _get_gamma_market(self, market_id: str) -> Dict[str, Any]:
        resp = self._client.get(f"{self._gamma_url}/markets/{market_id}")
        resp.raise_for_status()
        return resp.json()

    def _normalise(self, raw: Dict[str, Any]) -> Optional[MarketSnapshot]:
        """Convert a raw Gamma API response to a MarketSnapshot."""
        market_id = str(raw.get("id") or raw.get("conditionId") or "")
        if not market_id:
            return None

        # Extract token IDs
        tokens = raw.get("tokens", []) or raw.get("clobTokenIds", [])
        yes_token_id: Optional[str] = None
        no_token_id: Optional[str] = None
        if isinstance(tokens, list) and len(tokens) >= 2:
            yes_token_id = str(tokens[0]) if tokens[0] else None
            no_token_id = str(tokens[1]) if tokens[1] else None

        # Prices
        outcomes = raw.get("outcomePrices", [])
        if isinstance(outcomes, list) and len(outcomes) >= 2:
            try:
                yes_price = float(outcomes[0])
                no_price = float(outcomes[1])
            except (ValueError, TypeError):
                yes_price, no_price = 0.5, 0.5
        else:
            yes_price = float(raw.get("bestBid", 0.5))
            no_price = 1.0 - yes_price

        mid_price = (yes_price + no_price) / 2.0
        spread = abs(yes_price - (1.0 - no_price))

        # Dates
        resolution_date = _parse_date(raw.get("endDate") or raw.get("resolutionDate"))
        end_date = _parse_date(raw.get("endDate"))

        # Category / tags
        category = (raw.get("category") or raw.get("groupItemTitle") or "unknown").lower()
        tags = raw.get("tags", []) or []
        if isinstance(tags, list):
            tags = [str(t) for t in tags if t]

        return MarketSnapshot(
            market_id=market_id,
            question=raw.get("question", ""),
            category=category,
            yes_price=max(0.001, min(0.999, yes_price)),
            no_price=max(0.001, min(0.999, no_price)),
            mid_price=max(0.001, min(0.999, mid_price)),
            spread=max(0.0, spread),
            liquidity=float(raw.get("liquidity") or 0),
            volume=float(raw.get("volume") or raw.get("volumeNum") or 0),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            timestamp=datetime.now(timezone.utc),
            resolution_date=resolution_date,
            end_date=end_date,
            is_active=bool(raw.get("active", True)),
            slug=raw.get("slug"),
            tags=tags,
            raw_payload=raw,
        )


def _parse_date(val: Any) -> Optional[datetime]:
    """Parse a date string into a UTC datetime, return None on failure."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
