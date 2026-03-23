"""
AI trading assistant via Anthropic. Never passes secrets into prompts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from polymarket_alpha.config import get_config
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a professional quantitative trading assistant embedded in a fintech dashboard.
You explain risk, strategy, and market concepts clearly. You never invent live prices.
You remind users that trading involves loss of capital and that this software is not financial advice.
Never ask for or repeat API keys. Keep answers concise unless the user asks for detail."""


def build_context_block(safe_context: Dict[str, Any]) -> str:
    lines = [
        "[Runtime context — no secrets]",
        f"Simulation mode: {safe_context.get('simulation_mode', True)}",
        f"Kill switch: {safe_context.get('kill_switch', False)}",
        f"Autonomous: {safe_context.get('autonomous_mode', True)}",
        f"Risk preset: {safe_context.get('risk_level', 'medium')}",
        f"Selected market: {safe_context.get('selected_market', 'none')}",
        f"Equity (dashboard ledger): {safe_context.get('equity', 'n/a')}",
    ]
    return "\n".join(lines)


def chat_completion(
    user_message: str,
    history: List[Dict[str, str]],
    safe_context: Dict[str, Any],
) -> str:
    cfg = get_config()
    if not cfg.anthropic_api_key:
        return (
            "Anthropic API key is not configured. Set `ANTHROPIC_API_KEY` in your environment "
            "(see `.env.example`). I can still remind you: use paper/simulation mode until you "
            "understand the system; live trading requires explicit env gates and is high risk."
        )

    try:
        import anthropic
    except ImportError:
        return "The `anthropic` package is not installed."

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    ctx = build_context_block(safe_context)

    messages: List[Dict[str, str]] = []
    for turn in history[-12:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"{ctx}\n\nUser: {user_message}"})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if not messages:
        messages = [{"role": "user", "content": f"{ctx}\n\nUser: {user_message}"}]

    try:
        msg = client.messages.create(
            model=cfg.llm_model,
            max_tokens=min(cfg.llm_max_tokens, 2048),
            temperature=cfg.llm_temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        block = msg.content[0]
        if hasattr(block, "text"):
            return block.text
        return str(block)
    except Exception as exc:
        logger.error("chat_completion failed", extra={"error": str(exc)})
        return f"Assistant error: {exc!s}. Check model name and API key permissions."
