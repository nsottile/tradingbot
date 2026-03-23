"""Streamlit chat UI for the trading assistant."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from polymarket_alpha.services import trading_chat


def render_ai_chat(safe_context: Dict[str, Any]) -> None:
    st.subheader("AI Assistant")
    st.caption("Trading copilot — not financial advice. API keys are never sent to this chat.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Ask about risk controls, execution modes, or how to interpret the dashboard.",
            }
        ]

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Message the trading assistant…")
    if prompt:
        reply = trading_chat.chat_completion(
            prompt,
            st.session_state.chat_messages,
            safe_context,
        )
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
