from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIControlState:
    autonomous_mode: bool = True
    risk_level: str = "medium"  # low, medium, high
    capital_allocation_pct: float = 1.0
    strategy_name: str = "hybrid_default"
    interval_seconds: int = 60
    # True = always route to paper broker (simulation)
    simulation_mode: bool = True
    # User must confirm in UI before live routing (in addition to env gates)
    live_trading_confirmed: bool = False
    # Instant halt: no new orders; optional cancel on live venues
    kill_switch: bool = False

