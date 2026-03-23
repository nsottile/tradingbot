from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Dict, List

from polymarket_alpha.engine.ledger import PortfolioLedger


def compute_metrics(ledger: PortfolioLedger, equity_curve: List[float]) -> Dict[str, float]:
    sells = [t for t in ledger.trades if t.side == "SELL"]
    wins = [t for t in sells if t.realized_pnl > 0]
    returns = []
    for i in range(1, len(equity_curve)):
        prev = max(equity_curve[i - 1], 1e-9)
        returns.append((equity_curve[i] - equity_curve[i - 1]) / prev)

    peak = equity_curve[0] if equity_curve else ledger.initial_capital
    max_drawdown = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / max(peak, 1e-9)
        max_drawdown = max(max_drawdown, dd)

    sharpe = 0.0
    if len(returns) >= 2:
        avg = mean(returns)
        std = sqrt(sum((r - avg) ** 2 for r in returns) / (len(returns) - 1))
        if std > 0:
            sharpe = (avg / std) * sqrt(252)

    return {
        "equity": ledger.equity,
        "pnl": ledger.equity - ledger.initial_capital,
        "win_rate": (len(wins) / len(sells)) if sells else 0.0,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trades": float(len(ledger.trades)),
        "active_positions": float(len(ledger.positions)),
    }

