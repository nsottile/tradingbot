from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from polymarket_alpha.config import get_config
from polymarket_alpha.data.pipeline import UnifiedMarketDataPipeline
from polymarket_alpha.data.repository import TradingRepository
from polymarket_alpha.engine.analytics import compute_metrics
from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine
from polymarket_alpha.payments import stripe_scaffold
from polymarket_alpha.ui.chat_panel import render_ai_chat
from polymarket_alpha.ui.components import (
    inject_theme,
    metric_row_terminal,
    panel_title,
    terminal_header_row,
)


@st.cache_resource
def get_engine() -> AutonomousTradingEngine:
    return AutonomousTradingEngine()


@st.cache_resource
def get_data_pipeline() -> UnifiedMarketDataPipeline:
    return UnifiedMarketDataPipeline()


@st.cache_resource
def get_repo() -> TradingRepository:
    return TradingRepository()


def _decisions_df(hours: int = 72) -> pd.DataFrame:
    repo = get_repo()
    rows = repo.get_decisions(since=datetime.now(timezone.utc) - timedelta(hours=hours), limit=5000)
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _engine_summary(engine: AutonomousTradingEngine) -> Dict[str, object]:
    cfg = get_config()
    return {
        "simulation": engine.control.simulation_mode,
        "kill_switch": engine.control.kill_switch,
        "live_env_ok": bool(cfg.enable_live_trading and not cfg.paper_trading),
        "uses_paper": engine._router.uses_paper_path(),
        "autonomous": engine.control.autonomous_mode,
        "risk": engine.control.risk_level,
    }


def _safe_chat_context(engine: AutonomousTradingEngine) -> Dict[str, object]:
    return {
        "simulation_mode": engine.control.simulation_mode,
        "kill_switch": engine.control.kill_switch,
        "autonomous_mode": engine.control.autonomous_mode,
        "risk_level": engine.control.risk_level,
        "selected_market": st.session_state.get("selected_market", "none"),
        "equity": round(engine.ledger.equity, 2),
    }


def render_trade_terminal(engine: AutonomousTradingEngine) -> None:
    terminal_header_row(_engine_summary(engine))
    cfg = get_config()
    metrics = compute_metrics(engine.ledger, engine.equity_curve)
    metric_row_terminal(metrics)

    snap_df = _markets_dataframe()
    left, center, right = st.columns([1, 2.2, 1])

    with left:
        panel_title("Watchlist")
        if snap_df.empty:
            st.info("No markets.")
        else:
            options = snap_df["Symbol"].astype(str).tolist()[:80]
            sel = st.selectbox("Symbol", options, key="sym_sel")
            st.session_state["selected_market"] = sel
            sub = snap_df[snap_df["Symbol"].astype(str) == sel].head(1)
            if not sub.empty:
                st.metric("Last", f"{float(sub['Price'].iloc[0]):.4f}")
                st.caption(f"Vol {float(sub['Volume (24h)'].iloc[0]):,.0f}")

    with center:
        panel_title("Chart")
        fig = go.Figure()
        eq = engine.equity_curve[-500:] if engine.equity_curve else [cfg.initial_bankroll]
        fig.add_trace(
            go.Scatter(
                y=eq,
                mode="lines",
                name="Equity",
                line=dict(color="#3861fb", width=2),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#0b0e11",
            font_color="#eaecef",
            margin=dict(l=40, r=20, t=30, b=40),
            height=420,
            showlegend=True,
            xaxis=dict(showgrid=True, gridcolor="#1e2329"),
            yaxis=dict(showgrid=True, gridcolor="#1e2329"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        panel_title("Order ticket")
        st.caption("Manual orders are preview-only in this build; autonomous uses the router.")
        side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
        usd = st.number_input("Size (USD)", min_value=0.0, value=100.0, step=50.0)
        px_in = st.number_input("Limit price", min_value=0.0, value=0.5, step=0.01)
        if st.button("Preview risk check", use_container_width=True):
            st.success(f"{side} ${usd:,.2f} @ {px_in:.4f} — not sent (UI stub).")

    panel_title("Book / positions")
    t1, t2, t3 = st.tabs(["Positions", "Open orders", "AI stream"])
    with t1:
        rows: List[Dict] = []
        for pos in engine.ledger.positions.values():
            u = pos.unrealized_pnl
            rows.append(
                {
                    "Symbol": pos.symbol,
                    "Qty": round(pos.quantity, 6),
                    "Entry": round(pos.entry_price, 4),
                    "Mark": round(pos.current_price, 4),
                    "uPnL": round(u, 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
    with t2:
        st.caption("Live venue open orders require broker list endpoint wiring.")
        st.dataframe(pd.DataFrame(), use_container_width=True, height=180)
    with t3:
        for line in (engine.activity_feed[-20:] or ["No activity yet."])[::-1]:
            st.text(line)


def _markets_dataframe() -> pd.DataFrame:
    snapshots = get_data_pipeline().fetch_live()
    rows = [
        {
            "Source": s.source,
            "Asset Class": s.asset_class.value,
            "Venue": getattr(s, "venue", s.source),
            "Symbol": s.symbol,
            "Name": s.name[:72],
            "Price": s.price,
            "Spread": s.spread,
            "Volume (24h)": s.volume_24h,
            "Liquidity": s.liquidity,
            "Volatility": s.volatility_24h,
            "AI score": round(min(1.0, max(0.0, (s.volume_24h / (s.liquidity + 1)) * 0.1)), 4),
        }
        for s in snapshots
    ]
    return pd.DataFrame(rows)


def render_markets_explorer() -> None:
    st.subheader("Markets explorer")
    df = _markets_dataframe()
    if df.empty:
        st.info("No live markets.")
        return
    c1, c2, c3 = st.columns(3)
    cats = sorted(df["Asset Class"].unique())
    af = c1.multiselect("Asset class", cats, default=cats)
    sort_key = c2.selectbox("Sort", ["Volume (24h)", "Volatility", "Liquidity", "AI score", "Spread"])
    trend = c3.selectbox("View", ["All", "Trending (vol)", "Tight spread"])
    out = df[df["Asset Class"].isin(af)].sort_values(sort_key, ascending=False)
    if trend == "Trending (vol)":
        out = out.head(40)
    elif trend == "Tight spread":
        out = out.nsmallest(60, "Spread")
    st.dataframe(out, use_container_width=True, height=520)


def render_portfolio(engine: AutonomousTradingEngine) -> None:
    st.subheader("Portfolio")
    m = compute_metrics(engine.ledger, engine.equity_curve)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${m['equity']:,.2f}")
    c2.metric("Cash", f"${engine.ledger.cash:,.2f}")
    c3.metric("Open positions", int(m.get("active_positions", 0)))
    c4.metric("Total trades", int(m.get("trades", 0)))
    st.markdown("**Exposure** — by asset class (ledger does not tag venue per position in v1).")
    st.info("Wire position metadata from `NormalizedMarketSnapshot.asset_class` on open for full allocation charts.")


def render_trade_history() -> None:
    st.subheader("Trade history")
    df = _decisions_df(240)
    if df.empty:
        st.info("No history.")
        return
    q = st.text_input("Search market id")
    if q:
        df = df[df["market_id"].astype(str).str.contains(q, case=False, na=False)]
    cols = [c for c in ["timestamp", "market_id", "action", "position_size", "edge_after_costs", "confidence", "reason"] if c in df.columns]
    st.dataframe(df[cols].sort_values("timestamp", ascending=False), use_container_width=True, height=480)


def render_control_panel(engine: AutonomousTradingEngine) -> None:
    st.subheader("AI control")
    cfg = get_config()
    st.markdown(
        f"Env: `ENABLE_LIVE_TRADING={cfg.enable_live_trading}` · `PAPER_TRADING={cfg.paper_trading}` · "
        f"`MAX_ORDER_NOTIONAL_USD={cfg.max_order_notional_usd}`"
    )
    engine.control.autonomous_mode = st.toggle("Autonomous loop", value=engine.control.autonomous_mode)
    engine.control.simulation_mode = st.toggle(
        "Simulation (paper) execution",
        value=engine.control.simulation_mode,
        help="When on, all fills go through the paper broker.",
    )
    engine.control.kill_switch = st.toggle(
        "Kill switch — block new orders",
        value=engine.control.kill_switch,
    )
    if engine.control.kill_switch:
        if st.button("Cancel live open orders (Polymarket CLOB)"):
            engine._router.emergency_cancel_all_live()
            st.warning("Sent cancel request to broker stub.")

    engine.control.live_trading_confirmed = st.checkbox(
        "I confirm live routing may send real orders (requires env gates above)",
        value=engine.control.live_trading_confirmed,
    )

    engine.control.risk_level = st.selectbox(
        "Risk preset",
        ["low", "medium", "high"],
        index=["low", "medium", "high"].index(engine.control.risk_level),
    )
    engine.control.capital_allocation_pct = st.slider("Capital allocation", 0.1, 1.0, float(engine.control.capital_allocation_pct), 0.05)
    engine.control.strategy_name = st.selectbox("Strategy", ["hybrid_default"])
    engine.control.interval_seconds = st.selectbox("Loop interval (s)", [30, 60, 120, 300], index=1)

    if st.button("Run one autonomous cycle", type="primary"):
        with st.spinner("Running…"):
            engine.run_cycle()
        st.success("Cycle done.")


def render_billing() -> None:
    st.subheader("Billing (scaffold)")
    st.markdown(
        "Credits/subscriptions via Stripe require a **webhook endpoint** on HTTPS. "
        "Streamlit alone cannot receive webhooks — use FastAPI, Cloud Functions, or Next.js. See README."
    )
    if stripe_scaffold.stripe_configured():
        st.success("Stripe env vars detected.")
        if st.button("Create Checkout Session (test)"):
            r = stripe_scaffold.create_checkout_session_stub(
                success_url="https://example.com/success",
                cancel_url="https://example.com/cancel",
            )
            st.json(r)
    else:
        st.warning("Set `STRIPE_SECRET_KEY` and `STRIPE_PRICE_CREDITS_ID` to enable.")


def render_app() -> None:
    st.set_page_config(page_title="Alpha Terminal", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    engine = get_engine()
    st.markdown("<h1 style='font-size:1.4rem; margin:0; font-weight:600;'>Alpha Terminal</h1>", unsafe_allow_html=True)
    st.caption("Professional-grade dashboard — simulation default; live execution is gated.")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Trade",
            "Markets",
            "Portfolio",
            "History",
            "Control",
            "AI Chat",
            "Billing",
        ],
    )

    if page == "Trade":
        render_trade_terminal(engine)
    elif page == "Markets":
        render_markets_explorer()
    elif page == "Portfolio":
        render_portfolio(engine)
    elif page == "History":
        render_trade_history()
    elif page == "Control":
        render_control_panel(engine)
    elif page == "AI Chat":
        render_ai_chat(_safe_chat_context(engine))
    else:
        render_billing()
