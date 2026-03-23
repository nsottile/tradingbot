from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from polymarket_alpha.data.providers import PlaceholderProvider, PolymarketProvider
from polymarket_alpha.data.schemas import AssetClass, NormalizedMarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UnifiedMarketDataPipeline:
    """Centralized market ingestion across provider adapters."""

    def __init__(self) -> None:
        self._providers = [
            PolymarketProvider(),
            PlaceholderProvider("crypto_adapter", AssetClass.CRYPTO),
            PlaceholderProvider("stocks_adapter", AssetClass.STOCK),
            PlaceholderProvider("forex_adapter", AssetClass.FOREX),
        ]

    def fetch_live(self) -> List[NormalizedMarketSnapshot]:
        all_snapshots: List[NormalizedMarketSnapshot] = []
        counts: Dict[str, int] = defaultdict(int)
        for provider in self._providers:
            try:
                rows = provider.fetch_live()
                all_snapshots.extend(rows)
                counts[provider.source_name] += len(rows)
            except Exception as exc:
                logger.error(
                    "Provider fetch failed",
                    extra={"provider": provider.source_name, "error": str(exc)},
                )
        logger.info("Unified data fetched", extra={"provider_counts": dict(counts)})
        return all_snapshots

