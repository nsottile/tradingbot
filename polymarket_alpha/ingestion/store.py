"""
MarketStore: durable storage for market snapshots, trade decisions,
LLM signals, and calibration metrics using PostgreSQL.

Uses raw SQL via psycopg2 to keep dependencies minimal and queries explicit.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from polymarket_alpha.config import get_config
from polymarket_alpha.schemas import (
    CalibrationMetrics,
    LLMSignal,
    MarketSnapshot,
    TradeDecision,
)
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    market_id       TEXT NOT NULL,
    question        TEXT,
    category        TEXT,
    yes_price       DOUBLE PRECISION,
    no_price        DOUBLE PRECISION,
    mid_price       DOUBLE PRECISION,
    spread          DOUBLE PRECISION,
    liquidity       DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    timestamp       TIMESTAMPTZ NOT NULL,
    resolution_date TIMESTAMPTZ,
    is_active       BOOLEAN,
    raw_payload     JSONB
);
CREATE INDEX IF NOT EXISTS idx_ms_market_id ON market_snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_ms_timestamp  ON market_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS trade_decisions (
    id                BIGSERIAL PRIMARY KEY,
    market_id         TEXT NOT NULL,
    timestamp         TIMESTAMPTZ NOT NULL,
    signal_probs      JSONB,
    calibrated_prob   DOUBLE PRECISION,
    expected_value    DOUBLE PRECISION,
    edge_after_costs  DOUBLE PRECISION,
    confidence        DOUBLE PRECISION,
    position_size     DOUBLE PRECISION,
    action            TEXT,
    reason            TEXT,
    model_version     TEXT,
    data_version      TEXT,
    skipped           BOOLEAN,
    skip_reason       TEXT
);
CREATE INDEX IF NOT EXISTS idx_td_market_id ON trade_decisions(market_id);
CREATE INDEX IF NOT EXISTS idx_td_timestamp  ON trade_decisions(timestamp);

CREATE TABLE IF NOT EXISTS llm_signals (
    id                  BIGSERIAL PRIMARY KEY,
    market_id           TEXT NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    probability         DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    reasoning_summary   TEXT,
    bullish_factors     JSONB,
    bearish_factors     JSONB,
    uncertainty_sources JSONB,
    data_needed         JSONB,
    raw_prompt          TEXT,
    raw_response        TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_market_id ON llm_signals(market_id);

CREATE TABLE IF NOT EXISTS calibration_metrics (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    market_family   TEXT,
    brier_score     DOUBLE PRECISION,
    log_loss        DOUBLE PRECISION,
    n_samples       INTEGER,
    ece             DOUBLE PRECISION,
    overconfidence  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id                BIGSERIAL PRIMARY KEY,
    market_id         TEXT NOT NULL,
    entry_timestamp   TIMESTAMPTZ NOT NULL,
    exit_timestamp    TIMESTAMPTZ,
    action            TEXT,
    entry_price       DOUBLE PRECISION,
    exit_price        DOUBLE PRECISION,
    position_size     DOUBLE PRECISION,
    gross_pnl         DOUBLE PRECISION,
    fees              DOUBLE PRECISION,
    slippage          DOUBLE PRECISION,
    net_pnl           DOUBLE PRECISION,
    resolved          BOOLEAN,
    correct_direction BOOLEAN
);
"""


class MarketStore:
    """
    Persistent store for all system data.

    Falls back to in-memory lists when PostgreSQL is unavailable
    so the system can run without a database for testing.
    """

    def __init__(self) -> None:
        self._conn = None
        self._in_memory: Dict[str, List[Any]] = {
            "market_snapshots": [],
            "trade_decisions": [],
            "llm_signals": [],
            "calibration_metrics": [],
            "backtest_trades": [],
        }
        self._use_db = False

        if HAS_PSYCOPG2:
            self._try_connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        cfg = get_config()
        try:
            self._conn = psycopg2.connect(cfg.database_url)
            self._conn.autocommit = True
            self._run_ddl()
            self._use_db = True
            logger.info("Connected to PostgreSQL")
        except Exception as exc:
            logger.warning(
                "PostgreSQL unavailable, using in-memory store",
                extra={"error": str(exc)},
            )
            self._conn = None
            self._use_db = False

    def _run_ddl(self) -> None:
        if self._conn is None:
            return
        with self._conn.cursor() as cur:
            cur.execute(DDL)

    # ------------------------------------------------------------------
    # Market snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self, snap: MarketSnapshot) -> None:
        if self._use_db and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO market_snapshots
                        (market_id, question, category, yes_price, no_price,
                         mid_price, spread, liquidity, volume, timestamp,
                         resolution_date, is_active, raw_payload)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        snap.market_id,
                        snap.question,
                        snap.category,
                        snap.yes_price,
                        snap.no_price,
                        snap.mid_price,
                        snap.spread,
                        snap.liquidity,
                        snap.volume,
                        snap.timestamp,
                        snap.resolution_date,
                        snap.is_active,
                        json.dumps(snap.raw_payload),
                    ),
                )
        else:
            self._in_memory["market_snapshots"].append(snap.model_dump())

    def get_snapshots(
        self,
        market_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        if self._use_db and self._conn:
            conditions = []
            params: List[Any] = []
            if market_id:
                conditions.append("market_id = %s")
                params.append(market_id)
            if since:
                conditions.append("timestamp >= %s")
                params.append(since)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT * FROM market_snapshots {where} ORDER BY timestamp DESC LIMIT %s",
                    params,
                )
                return [dict(r) for r in cur.fetchall()]
        else:
            rows = self._in_memory["market_snapshots"]
            if market_id:
                rows = [r for r in rows if r.get("market_id") == market_id]
            return rows[-limit:]

    # ------------------------------------------------------------------
    # Trade decisions
    # ------------------------------------------------------------------

    def save_decision(self, decision: TradeDecision) -> None:
        row = decision.model_dump()
        if self._use_db and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trade_decisions
                        (market_id, timestamp, signal_probs, calibrated_prob,
                         expected_value, edge_after_costs, confidence,
                         position_size, action, reason, model_version,
                         data_version, skipped, skip_reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row["market_id"],
                        row["timestamp"],
                        json.dumps(row["signal_probs"]),
                        row["calibrated_prob"],
                        row["expected_value"],
                        row["edge_after_costs"],
                        row["confidence"],
                        row["position_size"],
                        row["action"],
                        row["reason"],
                        row["model_version"],
                        row["data_version"],
                        row["skipped"],
                        row["skip_reason"],
                    ),
                )
        else:
            self._in_memory["trade_decisions"].append(row)

    def get_decisions(
        self,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        if self._use_db and self._conn:
            params: List[Any] = []
            where = ""
            if since:
                where = "WHERE timestamp >= %s"
                params.append(since)
            params.append(limit)
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT * FROM trade_decisions {where} ORDER BY timestamp DESC LIMIT %s",
                    params,
                )
                return [dict(r) for r in cur.fetchall()]
        else:
            rows = self._in_memory["trade_decisions"]
            return rows[-limit:]

    # ------------------------------------------------------------------
    # LLM signals
    # ------------------------------------------------------------------

    def save_llm_signal(self, signal: LLMSignal) -> None:
        row = signal.model_dump()
        if self._use_db and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_signals
                        (market_id, timestamp, probability, confidence,
                         reasoning_summary, bullish_factors, bearish_factors,
                         uncertainty_sources, data_needed, raw_prompt, raw_response)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row["market_id"],
                        row["timestamp"],
                        row["probability"],
                        row["confidence"],
                        row["reasoning_summary"],
                        json.dumps(row["bullish_factors"]),
                        json.dumps(row["bearish_factors"]),
                        json.dumps(row["uncertainty_sources"]),
                        json.dumps(row["data_needed"]),
                        row["raw_prompt"],
                        row["raw_response"],
                    ),
                )
        else:
            self._in_memory["llm_signals"].append(row)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def save_calibration_metrics(self, metrics: CalibrationMetrics) -> None:
        row = metrics.model_dump()
        if self._use_db and self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO calibration_metrics
                        (timestamp, market_family, brier_score, log_loss,
                         n_samples, ece, overconfidence)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row["timestamp"],
                        row["market_family"],
                        row["brier_score"],
                        row["log_loss"],
                        row["n_samples"],
                        row["ece"],
                        row["overconfidence"],
                    ),
                )
        else:
            self._in_memory["calibration_metrics"].append(row)

    def get_calibration_history(
        self, market_family: str = "global", limit: int = 200
    ) -> List[Dict[str, Any]]:
        if self._use_db and self._conn:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM calibration_metrics
                    WHERE market_family = %s
                    ORDER BY timestamp DESC LIMIT %s
                    """,
                    (market_family, limit),
                )
                return [dict(r) for r in cur.fetchall()]
        else:
            return [
                r for r in self._in_memory["calibration_metrics"]
                if r.get("market_family") == market_family
            ][-limit:]
