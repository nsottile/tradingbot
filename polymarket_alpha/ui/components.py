"""Hyperliquid-inspired dark terminal theme and layout primitives."""

from __future__ import annotations

from typing import Dict

import streamlit as st

# Palette aligned with dense trading terminals (Hyperliquid-like)
HL_BG = "#0b0e11"
HL_PANEL = "#131722"
HL_BORDER = "#1e2329"
HL_TEXT = "#eaecef"
HL_MUTED = "#848e9c"
HL_GREEN = "#0ecb81"
HL_RED = "#f6465d"
HL_ACCENT = "#3861fb"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp, .main, [data-testid="stAppViewContainer"] {{
            background-color: {HL_BG} !important;
            color: {HL_TEXT};
        }}
        header[data-testid="stHeader"] {{ background-color: {HL_BG} !important; }}
        div[data-testid="stToolbar"] {{ background-color: {HL_BG} !important; }}
        .block-container {{ padding-top: 1rem; max-width: 100%; }}
        div[data-testid="stMetric"] {{
            background: {HL_PANEL};
            border: 1px solid {HL_BORDER};
            border-radius: 4px;
            padding: 10px 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        .hl-panel {{
            background: {HL_PANEL};
            border: 1px solid {HL_BORDER};
            border-radius: 4px;
            padding: 12px;
            margin-bottom: 8px;
        }}
        .hl-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {HL_MUTED};
            margin-bottom: 8px;
        }}
        .status-chip {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 6px;
            font-family: ui-monospace, monospace;
        }}
        .chip-on {{ background: #0d2818; color: {HL_GREEN}; border: 1px solid #1a4d2e; }}
        .chip-off {{ background: #2a1f0d; color: #f0b429; border: 1px solid #4a3a12; }}
        .chip-danger {{ background: #2a1215; color: {HL_RED}; border: 1px solid #5c1f26; }}
        .chip-info {{ background: #12182a; color: {HL_ACCENT}; border: 1px solid #243a6b; }}
        .pnl-pos {{ color: {HL_GREEN}; }}
        .pnl-neg {{ color: {HL_RED}; }}
        div[data-testid="stSidebar"] {{ background-color: {HL_PANEL} !important; border-right: 1px solid {HL_BORDER}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def terminal_header_row(engine_summary: Dict[str, object]) -> None:
    """Top status bar: mode, risk, paper/live path."""
    c = engine_summary
    sim = c.get("simulation", True)
    kill = c.get("kill_switch", False)
    live_ok = c.get("live_env_ok", False)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        st.markdown(
            f"<span class='status-chip {'chip-on' if sim else 'chip-danger'}'>"
            f"{'SIMULATION' if sim else 'LIVE ROUTING'}</span>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<span class='status-chip chip-info'>RISK: {str(c.get('risk', 'medium')).upper()}</span>",
            unsafe_allow_html=True,
        )
    with col3:
        if kill:
            st.markdown("<span class='status-chip chip-danger'>KILL SWITCH</span>", unsafe_allow_html=True)
        elif live_ok:
            st.markdown("<span class='status-chip chip-on'>LIVE ENV OK</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-chip chip-off'>LIVE LOCKED</span>", unsafe_allow_html=True)
    with col4:
        st.caption(f"Paper path: **{c.get('uses_paper', True)}** | Autonomous: **{c.get('autonomous', True)}**")


def metric_row_terminal(metrics: Dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    pnl = metrics.get("pnl", 0.0)
    pnl_s = f"${pnl:,.2f}"
    c1.metric("Equity", f"${metrics.get('equity', 0.0):,.2f}")
    c2.metric("PnL", pnl_s, delta=None)
    c3.metric("Win rate", f"{metrics.get('win_rate', 0.0):.1%}")
    c4.metric("Max DD", f"{metrics.get('max_drawdown', 0.0):.1%}")
    c5.metric("Sharpe", f"{metrics.get('sharpe', 0.0):.2f}")


def panel_title(title: str) -> None:
    st.markdown(f"<div class='hl-title'>{title}</div>", unsafe_allow_html=True)
