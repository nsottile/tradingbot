"""Shared types for broker integrations (no secrets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class BrokerVenue(str, Enum):
    PAPER = "paper"
    POLYMARKET = "polymarket"
    ALPACA = "alpaca"
    BINANCE = "binance"
    SPORTS = "sports"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    """Normalized order intent (validated before send)."""

    venue: BrokerVenue
    market_id: str
    symbol: str
    side: OrderSide
    notional_usd: float
    limit_price: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BalanceSnapshot:
    venue: BrokerVenue
    currency: str
    available: float
    as_of: datetime


@dataclass
class OpenOrder:
    order_id: str
    market_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
