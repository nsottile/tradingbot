from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from polymarket_alpha.data.pipeline import UnifiedMarketDataPipeline
from polymarket_alpha.data.repository import TradingRepository
from polymarket_alpha.engine.orchestrator import AutonomousTradingEngine
from polymarket_alpha.ui.components import inject_theme, metric_row, status_chips


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


def render_main_dashboard(engine: AutonomousTradingEngine) -> None:
    st.subheader("Main Dashboard")
    metrics = {
        "equity": engine.ledger.equity,
        "pnl": engine.ledger.equity - engine.ledger.initial_capital,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
    }
    metric_row(metrics)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Portfolio Curve")
        curve_df = pd.DataFrame({"equity": engine.equity_curve})
        curve_df.index.name = "step"
        st.line_chart(curve_df)
    with c2:
        st.markdown("#### AI Activity Feed")
        feed = engine.activity_feed[-15:] or ["No recent AI actions."]
        for item in reversed(feed):
            st.caption(item)

    st.markdown("#### Active Positions")
    rows: List[Dict] = []
    for pos in engine.ledger.positions.values():
        rows.append(
            {
                "Symbol": pos.symbol,
                "Qty": pos.quantity,
                "Entry": pos.entry_price,
                "Current": pos.current_price,
                "Unrealized PnL": pos.unrealized_pnl,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_markets() -> None:
    st.subheader("Markets Scanner")
    snapshots = get_data_pipeline().fetch_live()
    rows = [
        {
            "Source": s.source,
            "Asset Class": s.asset_class.value,
            "Symbol": s.symbol,
            "Name": s.name[:80],
            "Price": s.price,
            "Spread": s.spread,
            "Volume (24h)": s.volume_24h,
            "Liquidity": s.liquidity,
            "Volatility": s.volatility_24h,
        }
        for s in snapshots
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No live markets available.")
        return

    col1, col2 = st.columns(2)
    asset_filter = col1.multiselect("Asset Class", sorted(df["Asset Class"].unique()), default=sorted(df["Asset Class"].unique()))
    sort_key = col2.selectbox("Sort By", ["Volume (24h)", "Volatility", "Spread", "Liquidity"])
    filtered = df[df["Asset Class"].isin(asset_filter)].sort_values(sort_key, ascending=False)
    st.dataframe(filtered, use_container_width=True, height=420)


def render_trade_history() -> None:
    st.subheader("Trade History")
    df = _decisions_df(240)
    if df.empty:
        st.info("No trade history yet.")
        return

    search = st.text_input("Search market id")
    if search:
        df = df[df["market_id"].astype(str).str.contains(search, case=False, na=False)]
    if "action" in df.columns:
        action_filter = st.multiselect("Action Filter", sorted(df["action"].dropna().unique()), default=sorted(df["action"].dropna().unique()))
        df = df[df["action"].isin(action_filter)]
    cols = [c for c in ["timestamp", "market_id", "action", "position_size", "edge_after_costs", "confidence", "reason"] if c in df.columns]
    st.dataframe(df[cols].sort_values("timestamp", ascending=False), use_container_width=True, height=420)

    if "edge_after_costs" in df.columns and "action" in df.columns:
        pnl_fig = px.histogram(df, x="edge_after_costs", color="action", title="Trade Edge Distribution")
        st.plotly_chart(pnl_fig, use_container_width=True)


def render_ai_control_panel(engine: AutonomousTradingEngine) -> None:
    st.subheader("AI Control Panel")
    c1, c2, c3 = st.columns(3)
    engine.control.autonomous_mode = c1.toggle("Autonomous Mode", value=engine.control.autonomous_mode)
    engine.control.risk_level = c2.selectbox("Risk Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(engine.control.risk_level))
    engine.control.capital_allocation_pct = c3.slider("Capital Allocation", 0.1, 1.0, float(engine.control.capital_allocation_pct), 0.05)

    c4, c5 = st.columns(2)
    engine.control.strategy_name = c4.selectbox("Strategy", ["hybrid_default"])
    engine.control.interval_seconds = c5.selectbox("Loop Interval (sec)", [30, 60, 120, 300], index=1)

    st.markdown("#### Runtime Actions")
    run_once = st.button("Run One Autonomous Cycle", type="primary")
    if run_once:
        metrics = engine.run_cycle()
        st.success("Cycle complete.")
        metric_row(metrics)


def render_app() -> None:
    st.set_page_config(page_title="Autonomous AI Trading Platform", page_icon="📈", layout="wide")
    inject_theme()
    engine = get_engine()
    status_chips(engine.control.autonomous_mode, engine.control.risk_level)
    st.title("Autonomous AI Trading Simulation Platform")

    page = st.sidebar.radio("Navigate", ["Main Dashboard", "Markets", "Trade History", "AI Control Panel"])
    if page == "Main Dashboard":
        render_main_dashboard(engine)
    elif page == "Markets":
        render_markets()
    elif page == "Trade History":
        render_trade_history()
    else:
        render_ai_control_panel(engine)

