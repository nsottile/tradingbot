from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIControlState:
    autonomous_mode: bool = True
    risk_level: str = "medium"  # low, medium, high
    capital_allocation_pct: float = 1.0
    strategy_name: str = "hybrid_default"
    interval_seconds: int = 60

