from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    PREDICTION = "prediction"


class NormalizedMarketSnapshot(BaseModel):
    """Cross-asset normalized market record used by the new engine."""

    market_id: str
    symbol: str
    source: str
    asset_class: AssetClass
    venue: str = "polymarket"  # routing key for ExecutionRouter
    name: str
    category: str = "unknown"
    price: float = Field(ge=0.0)
    bid: Optional[float] = Field(default=None, ge=0.0)
    ask: Optional[float] = Field(default=None, ge=0.0)
    spread: float = Field(ge=0.0, default=0.0)
    volume_24h: float = Field(ge=0.0, default=0.0)
    volatility_24h: float = Field(ge=0.0, default=0.0)
    liquidity: float = Field(ge=0.0, default=0.0)
    sentiment_score: Optional[float] = None
    base_quote: Optional[str] = None  # e.g. BTCUSDT for crypto
    tick_size: Optional[float] = None
    timestamp: datetime
    raw_payload: Dict[str, Any] = Field(default_factory=dict)

