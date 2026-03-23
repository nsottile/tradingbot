from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from polymarket_alpha.data.schemas import AssetClass, NormalizedMarketSnapshot
from polymarket_alpha.ingestion.collector import MarketCollector
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseMarketProvider(ABC):
    source_name: str
    asset_class: AssetClass

    @abstractmethod
    def fetch_live(self) -> List[NormalizedMarketSnapshot]:
        raise NotImplementedError


class PolymarketProvider(BaseMarketProvider):
    source_name = "polymarket"
    asset_class = AssetClass.PREDICTION

    def __init__(self) -> None:
        self._collector = MarketCollector()

    def fetch_live(self) -> List[NormalizedMarketSnapshot]:
        try:
            snapshots = self._collector.fetch_active_markets()
        except Exception as exc:
            logger.error("Polymarket provider fetch failed", extra={"error": str(exc)})
            return []
        out: List[NormalizedMarketSnapshot] = []
        for snap in snapshots:
            price = max(0.0, min(1.0, snap.mid_price))
            payload = dict(snap.raw_payload)
            payload["_yes_token_id"] = snap.yes_token_id
            payload["_no_token_id"] = snap.no_token_id
            payload["_yes_price"] = snap.yes_price
            payload["_no_price"] = snap.no_price
            out.append(
                NormalizedMarketSnapshot(
                    market_id=snap.market_id,
                    symbol=snap.market_id,
                    source=self.source_name,
                    asset_class=self.asset_class,
                    venue="polymarket",
                    name=snap.question,
                    category=snap.category,
                    price=price,
                    bid=max(0.0, snap.yes_price - (snap.spread / 2.0)),
                    ask=min(1.0, snap.yes_price + (snap.spread / 2.0)),
                    spread=snap.spread,
                    volume_24h=snap.volume,
                    liquidity=snap.liquidity,
                    timestamp=snap.timestamp,
                    raw_payload=payload,
                )
            )
        return out


class PlaceholderProvider(BaseMarketProvider):
    """Stub provider to make the architecture multi-asset ready."""

    def __init__(self, source_name: str, asset_class: AssetClass) -> None:
        self.source_name = source_name
        self.asset_class = asset_class

    def fetch_live(self) -> List[NormalizedMarketSnapshot]:
        return []

