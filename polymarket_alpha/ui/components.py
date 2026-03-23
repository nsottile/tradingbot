from __future__ import annotations

from typing import Dict

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .main { background-color: #0b1220; color: #e5e7eb; }
        div[data-testid="stMetric"] { background: #111a2e; border: 1px solid #1f2a44; padding: 12px; border-radius: 10px; }
        .status-chip { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; margin-right: 6px; }
        .status-live { background:#123821; color:#57f59e; }
        .status-warn { background:#3f2f13; color:#ffce6b; }
        .status-danger { background:#431a24; color:#ff7c95; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: Dict[str, float]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", f"${metrics.get('equity', 0.0):,.2f}")
    c2.metric("PnL", f"${metrics.get('pnl', 0.0):,.2f}")
    c3.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.1%}")
    c4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.1%}")
    c5.metric("Sharpe", f"{metrics.get('sharpe', 0.0):.2f}")


def status_chips(autonomous: bool, risk_level: str) -> None:
    mode_class = "status-live" if autonomous else "status-warn"
    st.markdown(
        f"<span class='status-chip {mode_class}'>{'AUTONOMOUS ON' if autonomous else 'AUTONOMOUS OFF'}</span>"
        f"<span class='status-chip status-live'>RISK: {risk_level.upper()}</span>",
        unsafe_allow_html=True,
    )

