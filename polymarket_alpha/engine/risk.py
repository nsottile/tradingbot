from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskProfile:
    name: str
    max_position_pct: float
    stop_loss_pct: float


RISK_PROFILES = {
    "low": RiskProfile(name="low", max_position_pct=0.02, stop_loss_pct=0.05),
    "medium": RiskProfile(name="medium", max_position_pct=0.05, stop_loss_pct=0.08),
    "high": RiskProfile(name="high", max_position_pct=0.10, stop_loss_pct=0.12),
}


def position_size(capital: float, confidence: float, risk_level: str) -> float:
    profile = RISK_PROFILES.get(risk_level, RISK_PROFILES["medium"])
    return max(0.0, capital * profile.max_position_pct * confidence)

