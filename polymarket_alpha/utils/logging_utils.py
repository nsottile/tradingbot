"""
Structured logging utilities.
Every decision log includes raw inputs, features, and final output
so that any trade can be reproduced from stored data.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in {
                "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "name",
                "message",
            }:
                continue
            payload[key] = val
        return json.dumps(payload, default=str)


def get_logger(name: str, level: str = "INFO", json_logs: bool = True) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name:
        Logger name (usually __name__).
    level:
        Log level string.
    json_logs:
        If True, emit JSON lines. Otherwise use plain text.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def log_decision(
    logger: logging.Logger,
    market_id: str,
    action: str,
    reason: str,
    features: Optional[Dict[str, Any]] = None,
    signal_probs: Optional[Dict[str, Any]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a structured decision log entry.
    Every trade decision must call this so the audit trail is complete.
    """
    payload: Dict[str, Any] = {
        "event": "trade_decision",
        "market_id": market_id,
        "action": action,
        "reason": reason,
    }
    if features:
        payload["features"] = features
    if signal_probs:
        payload["signal_probs"] = signal_probs
    if risk_state:
        payload["risk_state"] = risk_state
    if extra:
        payload.update(extra)

    logger.info("trade_decision", extra=payload)
