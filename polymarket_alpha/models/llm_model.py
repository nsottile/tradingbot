"""
LLMProbabilityModel: Claude-based probability signal.

Claude is used as ONE signal among many, not the sole decision-maker.
Every prompt and response is stored for audit and model evaluation.
Malformed responses are rejected; the signal is simply omitted.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import anthropic
from pydantic import ValidationError

from polymarket_alpha.config import get_config
from polymarket_alpha.schemas import LLMSignal, MarketSnapshot
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

# JSON schema Claude must return
_RESPONSE_SCHEMA = {
    "probability": "float in [0, 1] — your calibrated YES probability",
    "confidence": "float in [0, 1] — your confidence in this estimate",
    "reasoning_summary": "string — 1-3 sentence summary",
    "bullish_factors": ["list of strings — reasons YES is more likely"],
    "bearish_factors": ["list of strings — reasons NO is more likely"],
    "uncertainty_sources": ["list of strings — main sources of uncertainty"],
    "data_needed": ["list of strings — data that would most improve confidence"],
}

_SYSTEM_PROMPT = """You are a calibrated forecasting model. Your sole job is to estimate
the probability that a Polymarket prediction market resolves YES.

Rules:
- Return ONLY valid JSON. No prose before or after.
- probability must be a float in [0.0, 1.0].
- confidence must be a float in [0.0, 1.0] — how sure you are of your probability estimate.
- If you are genuinely uncertain, return a probability near 0.5 and a low confidence score.
- Do not round to exactly 0 or 1 unless you are certain.
- Base your answer on the information provided. Do not fabricate facts.
- Acknowledge uncertainty explicitly in uncertainty_sources.

Return JSON matching exactly this schema:
""" + json.dumps(_RESPONSE_SCHEMA, indent=2)


def _build_prompt(snap: MarketSnapshot, features: Optional[Dict[str, Any]] = None) -> str:
    """Construct the user prompt from a market snapshot."""
    lines = [
        f"Market question: {snap.question}",
        f"Category: {snap.category}",
        f"Current YES price: {snap.yes_price:.3f}",
        f"Current NO price: {snap.no_price:.3f}",
        f"Spread: {snap.spread:.4f}",
        f"Liquidity (USD): {snap.liquidity:.0f}",
        f"24h Volume (USD): {snap.volume:.0f}",
        f"Snapshot time (UTC): {snap.timestamp.isoformat()}",
    ]
    if snap.resolution_date:
        lines.append(f"Resolution date: {snap.resolution_date.isoformat()}")
    if snap.tags:
        lines.append(f"Tags: {', '.join(snap.tags)}")
    if features:
        selected = {
            k: round(v, 4)
            for k, v in features.items()
            if k in {
                "momentum_24h", "momentum_7d", "price_volatility_24h",
                "ttl_days", "log_liquidity", "spread_over_price",
            }
        }
        if selected:
            lines.append(f"Computed features: {json.dumps(selected)}")

    lines.append("\nReturn your JSON probability estimate now.")
    return "\n".join(lines)


class LLMProbabilityModel:
    """
    Queries Claude for a calibrated probability estimate.

    Returns an LLMSignal on success, None on failure.
    Failures are logged but never raise — the ensemble degrades gracefully.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        if cfg.anthropic_api_key:
            self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        else:
            self._client = None
            logger.warning("ANTHROPIC_API_KEY not set — LLM signal disabled")

    def predict(
        self,
        snap: MarketSnapshot,
        features: Optional[Dict[str, Any]] = None,
    ) -> Optional[LLMSignal]:
        """
        Query Claude and return a validated LLMSignal.
        Returns None if LLM is disabled, times out, or returns invalid JSON.
        """
        if self._client is None or not self._cfg.llm_enabled:
            return None

        prompt = _build_prompt(snap, features)
        raw_response = ""

        for attempt in range(self._cfg.llm_retry_attempts):
            try:
                message = self._client.messages.create(
                    model=self._cfg.llm_model,
                    max_tokens=self._cfg.llm_max_tokens,
                    temperature=self._cfg.llm_temperature,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_response = message.content[0].text
                break
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"LLM rate limit, waiting {wait}s", extra={"attempt": attempt})
                time.sleep(wait)
            except anthropic.APITimeoutError:
                logger.warning("LLM timeout", extra={"attempt": attempt, "market_id": snap.market_id})
                return None
            except Exception as exc:
                logger.error(
                    "LLM API error",
                    extra={"error": str(exc), "market_id": snap.market_id},
                )
                return None

        if not raw_response:
            return None

        parsed = self._parse_response(raw_response)
        if parsed is None:
            logger.warning(
                "LLM response parse failed",
                extra={"market_id": snap.market_id, "raw": raw_response[:500]},
            )
            return None

        try:
            signal = LLMSignal(
                market_id=snap.market_id,
                timestamp=datetime.now(timezone.utc),
                probability=parsed["probability"],
                confidence=parsed["confidence"],
                reasoning_summary=parsed.get("reasoning_summary", ""),
                bullish_factors=parsed.get("bullish_factors", []),
                bearish_factors=parsed.get("bearish_factors", []),
                uncertainty_sources=parsed.get("uncertainty_sources", []),
                data_needed=parsed.get("data_needed", []),
                raw_prompt=prompt,
                raw_response=raw_response,
            )
            return signal
        except ValidationError as exc:
            logger.warning(
                "LLMSignal validation failed",
                extra={"error": str(exc), "market_id": snap.market_id},
            )
            return None

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON from Claude's response.
        Strips markdown code fences if present.
        """
        cleaned = text.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object within the text
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                try:
                    data = json.loads(cleaned[start: end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        required = {"probability", "confidence"}
        if not required.issubset(data.keys()):
            return None

        # Coerce types
        try:
            data["probability"] = float(data["probability"])
            data["confidence"] = float(data["confidence"])
        except (TypeError, ValueError):
            return None

        # Clamp
        data["probability"] = max(0.01, min(0.99, data["probability"]))
        data["confidence"] = max(0.0, min(1.0, data["confidence"]))

        return data
