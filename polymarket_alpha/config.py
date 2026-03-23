"""
Configuration management for polymarket_alpha.
All settings are read from environment variables or a config file.
Credentials are NEVER hardcoded.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover
    # Fallback: plain BaseModel so the package loads without pydantic-settings installed.
    # Environment variable overrides will not work in this mode.
    from pydantic import BaseModel as BaseSettings  # type: ignore


class Config(BaseSettings):
    """
    Central configuration.
    Override any value via environment variable with the same name (uppercased).
    """

    # --- Polymarket API ---
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_api_key: Optional[str] = Field(default=None, env="POLYMARKET_API_KEY")
    polymarket_api_secret: Optional[str] = Field(default=None, env="POLYMARKET_API_SECRET")
    polymarket_api_passphrase: Optional[str] = Field(default=None, env="POLYMARKET_API_PASSPHRASE")

    # --- Anthropic ---
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    llm_model: str = "claude-opus-4-5"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0  # Deterministic for reproducibility

    # --- Database ---
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/polymarket_alpha",
        env="DATABASE_URL",
    )

    # --- Redis (optional) ---
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")

    # --- Collection ---
    poll_interval_seconds: int = 60
    max_markets_per_poll: int = 500
    stale_data_threshold_seconds: int = 300  # 5 min

    # --- Signal ---
    llm_enabled: bool = True
    llm_timeout_seconds: int = 30
    llm_retry_attempts: int = 2

    # --- Edge / Trading ---
    edge_threshold: float = 0.04          # Minimum edge after costs to trade
    kelly_fraction: float = 0.25          # Fractional Kelly (never full Kelly)
    max_position_pct: float = 0.05        # Max 5% of bankroll per position
    max_category_exposure_pct: float = 0.20  # Max 20% in one category
    max_daily_loss_pct: float = 0.05      # Halt if daily loss exceeds 5%
    max_drawdown_pct: float = 0.15        # Halt if drawdown exceeds 15%
    min_liquidity: float = 500.0          # Min liquidity in USD
    max_spread: float = 0.08             # Max acceptable spread
    slippage_estimate: float = 0.005      # Estimated slippage per trade
    fee_estimate: float = 0.02           # Polymarket fee ~2%
    min_confidence: float = 0.50         # Min model confidence to trade

    # --- Execution ---
    paper_trading: bool = True            # ALWAYS start in paper mode
    initial_bankroll: float = 10_000.0
    # Master env gate: live order routing disabled unless true (in addition to UI + paper_trading=false)
    enable_live_trading: bool = Field(default=False, env="ENABLE_LIVE_TRADING")
    max_order_notional_usd: float = Field(default=5_000.0, env="MAX_ORDER_NOTIONAL_USD")

    # --- Alpaca (stocks) ---
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_api_secret: Optional[str] = Field(default=None, env="ALPACA_API_SECRET")
    alpaca_base_url: Optional[str] = Field(default=None, env="ALPACA_BASE_URL")

    # --- Binance (crypto stub) ---
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    binance_rest_url: Optional[str] = Field(default=None, env="BINANCE_REST_URL")

    # --- Stripe (billing scaffold; no secrets in UI) ---
    stripe_secret_key: Optional[str] = Field(default=None, env="STRIPE_SECRET_KEY")
    stripe_webhook_secret: Optional[str] = Field(default=None, env="STRIPE_WEBHOOK_SECRET")
    stripe_price_credits_id: Optional[str] = Field(default=None, env="STRIPE_PRICE_CREDITS_ID")

    # --- Calibration ---
    calibration_min_samples: int = 30
    calibration_drift_threshold: float = 0.05  # Brier score degradation

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- Backtest ---
    backtest_start_date: str = "2023-01-01"
    backtest_end_date: str = "2024-01-01"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
