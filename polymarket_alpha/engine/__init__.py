"""Autonomous runtime engine modules.

Avoid eager imports here to prevent circular imports with `brokers` ↔ `engine.router`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AutonomousTradingEngine"]

if TYPE_CHECKING:
    from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine as AutonomousTradingEngineType


def __getattr__(name: str) -> Any:
    if name == "AutonomousTradingEngine":
        from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine

        return AutonomousTradingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
