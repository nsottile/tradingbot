from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class Position:
    market_id: str
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    opened_at: datetime

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity


@dataclass
class TradeRecord:
    market_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    realized_pnl: float = 0.0


@dataclass
class PortfolioLedger:
    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def execute_buy(self, market_id: str, symbol: str, notional: float, price: float) -> None:
        if notional <= 0 or price <= 0:
            return
        notional = min(notional, self.cash)
        qty = notional / price
        self.cash -= notional
        existing = self.positions.get(market_id)
        if existing:
            total_qty = existing.quantity + qty
            avg = ((existing.entry_price * existing.quantity) + (price * qty)) / max(total_qty, 1e-9)
            existing.quantity = total_qty
            existing.entry_price = avg
            existing.current_price = price
        else:
            self.positions[market_id] = Position(
                market_id=market_id,
                symbol=symbol,
                quantity=qty,
                entry_price=price,
                current_price=price,
                opened_at=datetime.now(timezone.utc),
            )
        self.trades.append(
            TradeRecord(
                market_id=market_id,
                symbol=symbol,
                side="BUY",
                quantity=qty,
                price=price,
                timestamp=datetime.now(timezone.utc),
            )
        )

    def mark_price(self, market_id: str, price: float) -> None:
        if market_id in self.positions:
            self.positions[market_id].current_price = price

    def maybe_stop_loss(self, market_id: str, stop_loss_pct: float) -> None:
        pos = self.positions.get(market_id)
        if not pos:
            return
        if pos.entry_price <= 0:
            return
        dd = (pos.current_price - pos.entry_price) / pos.entry_price
        if dd <= -abs(stop_loss_pct):
            self.close_position(market_id)

    def close_position(self, market_id: str) -> None:
        pos = self.positions.get(market_id)
        if not pos:
            return
        proceeds = pos.quantity * pos.current_price
        basis = pos.quantity * pos.entry_price
        realized = proceeds - basis
        self.cash += proceeds
        self.trades.append(
            TradeRecord(
                market_id=market_id,
                symbol=pos.symbol,
                side="SELL",
                quantity=pos.quantity,
                price=pos.current_price,
                timestamp=datetime.now(timezone.utc),
                realized_pnl=realized,
            )
        )
        del self.positions[market_id]

    @property
    def equity(self) -> float:
        return self.cash + sum(p.quantity * p.current_price for p in self.positions.values())

