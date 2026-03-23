"""
Core typed schemas for polymarket_alpha.
All data flowing through the system must conform to these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Action(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SKIP = "SKIP"


class MarketSnapshot(BaseModel):
    """Immutable snapshot of a Polymarket market at a point in time."""

    market_id: str
    question: str
    category: str = "unknown"
    yes_price: float = Field(ge=0.0, le=1.0)
    no_price: float = Field(ge=0.0, le=1.0)
    mid_price: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    liquidity: float = Field(ge=0.0)
    volume: float = Field(ge=0.0)
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    timestamp: datetime
    resolution_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_active: bool = True
    slug: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("yes_price", "no_price", "mid_price", mode="before")
    @classmethod
    def clamp_price(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))

    model_config = ConfigDict(frozen=True)


class SignalProbabilities(BaseModel):
    """Probability estimates from each signal source."""

    llm: Optional[float] = None
    microstructure: Optional[float] = None
    heuristic: Optional[float] = None
    ensemble: Optional[float] = None

    model_config = ConfigDict(frozen=True)


class TradeDecision(BaseModel):
    """Full record of a trade decision, regardless of whether it was executed."""

    market_id: str
    timestamp: datetime
    signal_probs: SignalProbabilities
    calibrated_prob: float
    expected_value: float
    edge_after_costs: float
    confidence: float = Field(ge=0.0, le=1.0)
    position_size: float = Field(ge=0.0)
    action: Action
    reason: str
    model_version: str = "unknown"
    data_version: str = "unknown"
    skipped: bool = False
    skip_reason: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class LLMSignal(BaseModel):
    """Validated output from the Claude signal layer."""

    market_id: str
    timestamp: datetime
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = ""
    bullish_factors: List[str] = Field(default_factory=list)
    bearish_factors: List[str] = Field(default_factory=list)
    uncertainty_sources: List[str] = Field(default_factory=list)
    data_needed: List[str] = Field(default_factory=list)
    raw_prompt: str = ""
    raw_response: str = ""

    model_config = ConfigDict(frozen=True)


class BacktestTrade(BaseModel):
    """A simulated trade in the backtest engine."""

    market_id: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime] = None
    action: Action
    entry_price: float
    exit_price: Optional[float] = None
    position_size: float
    gross_pnl: Optional[float] = None
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: Optional[float] = None
    resolved: bool = False
    correct_direction: Optional[bool] = None

    model_config = ConfigDict(frozen=True)


class CalibrationMetrics(BaseModel):
    """Calibration evaluation results."""

    timestamp: datetime
    market_family: str = "global"
    brier_score: float
    log_loss: float
    n_samples: int
    ece: float  # Expected Calibration Error
    overconfidence: float  # mean predicted - mean actual

    model_config = ConfigDict(frozen=True)


class RiskState(BaseModel):
    """Current risk state of the system."""

    timestamp: datetime
    bankroll: float
    daily_pnl: float
    daily_loss_limit: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    category_exposure: Dict[str, float] = Field(default_factory=dict)
    active_positions: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None

    model_config = ConfigDict(frozen=True)
