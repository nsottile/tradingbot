"""
DashboardApp: Streamlit dashboard for monitoring the trading system.

Shows:
- Live markets ranked by edge
- Recent trades and skip log
- Bankroll curve and drawdown
- Calibration curves and Brier score
- Category-level performance
- Signal breakdown per market
- Calibration drift alerts
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Ensure package imports resolve even if Streamlit is launched
# from inside the inner `polymarket_alpha/` package directory.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from polymarket_alpha.ingestion.collector import MarketCollector
    from polymarket_alpha.ingestion.store import MarketStore
    from polymarket_alpha.utils.logging_utils import get_logger
except ModuleNotFoundError:
    # Fallback when an installed package shadows local source.
    _PKG_ROOT = _THIS_FILE.parents[1]
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))
    MarketCollector = import_module("ingestion.collector").MarketCollector
    MarketStore = import_module("ingestion.store").MarketStore
    get_logger = import_module("utils.logging_utils").get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Polymarket Alpha",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


@st.cache_resource
def get_store() -> MarketStore:
    return MarketStore()


@st.cache_resource
def get_collector() -> MarketCollector:
    return MarketCollector()


def _load_decisions(store: MarketStore, hours: int = 24) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = store.get_decisions(since=since, limit=1000)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _load_calibration(store: MarketStore, family: str = "global") -> pd.DataFrame:
    rows = store.get_calibration_history(market_family=family, limit=200)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    lookback_hours = st.slider("Lookback (hours)", 1, 168, 24)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if auto_refresh:
        import time
        st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(1)
        st.rerun()

    st.divider()
    st.markdown("**System Mode**")
    try:
        from polymarket_alpha.config import get_config
        cfg = get_config()
        mode = "🟡 Paper Trading" if cfg.paper_trading else "🔴 LIVE TRADING"
        st.markdown(f"### {mode}")
        st.caption(f"Edge threshold: {cfg.edge_threshold:.1%}")
        st.caption(f"Kelly fraction: {cfg.kelly_fraction:.0%}")
        st.caption(f"Max position: {cfg.max_position_pct:.0%} bankroll")
    except Exception:
        st.warning("Config unavailable")

# -----------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------

st.title("📊 Polymarket Alpha Dashboard")

store = get_store()
decisions_df = _load_decisions(store, lookback_hours)
cal_df = _load_calibration(store)

# -----------------------------------------------------------------------
# Top KPI row
# -----------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

if not decisions_df.empty:
    trades = decisions_df[~decisions_df.get("skipped", pd.Series(True, index=decisions_df.index))]
    skips = decisions_df[decisions_df.get("skipped", pd.Series(False, index=decisions_df.index))]

    col1.metric("Total Decisions", len(decisions_df))
    col2.metric("Trades", len(trades))
    col3.metric("Skips", len(skips))

    if "edge_after_costs" in decisions_df.columns and len(trades) > 0:
        avg_edge = trades["edge_after_costs"].mean()
        col4.metric("Avg Edge", f"{avg_edge:.3f}")
    else:
        col4.metric("Avg Edge", "—")

    if not cal_df.empty and "brier_score" in cal_df.columns:
        last_brier = cal_df["brier_score"].iloc[-1]
        col5.metric("Last Brier Score", f"{last_brier:.4f}")
    else:
        col5.metric("Brier Score", "—")
else:
    for c, label in zip([col1, col2, col3, col4, col5],
                        ["Total Decisions", "Trades", "Skips", "Avg Edge", "Brier"]):
        c.metric(label, "—")

st.divider()

# -----------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------

tab_markets, tab_trades, tab_bankroll, tab_calibration, tab_signals = st.tabs([
    "🏪 Live Markets",
    "📋 Trade Log",
    "💰 Bankroll & Drawdown",
    "🎯 Calibration",
    "🔍 Signal Breakdown",
])

# -----------------------------------------------------------------------
# Tab 1: Live markets ranked by edge
# -----------------------------------------------------------------------

with tab_markets:
    st.subheader("Live Markets — Ranked by Expected Edge")

    try:
        collector = get_collector()
        with st.spinner("Fetching live markets…"):
            snapshots = collector.fetch_active_markets()

        if snapshots:
            rows = []
            for s in snapshots:
                rows.append({
                    "Market": s.question[:80],
                    "Category": s.category,
                    "YES Price": f"{s.yes_price:.3f}",
                    "NO Price": f"{s.no_price:.3f}",
                    "Spread": f"{s.spread:.4f}",
                    "Liquidity": f"${s.liquidity:,.0f}",
                    "Volume": f"${s.volume:,.0f}",
                    "Resolution": s.resolution_date.strftime("%Y-%m-%d") if s.resolution_date else "—",
                    "market_id": s.market_id,
                })
            market_df = pd.DataFrame(rows)

            # Filter controls
            cat_filter = st.multiselect(
                "Filter by category",
                options=sorted(market_df["Category"].unique()),
                default=[],
            )
            if cat_filter:
                market_df = market_df[market_df["Category"].isin(cat_filter)]

            min_liq = st.slider("Min liquidity ($)", 0, 50_000, 500, step=500)
            market_df_filtered = market_df[
                market_df["Liquidity"].str.replace("[$,]", "", regex=True).astype(float) >= min_liq
            ]

            st.dataframe(
                market_df_filtered.drop(columns=["market_id"]),
                use_container_width=True,
                height=400,
            )
            st.caption(f"{len(market_df_filtered)} markets shown")
        else:
            st.info("No active markets found. Check API connectivity.")

    except Exception as exc:
        st.error(f"Failed to fetch markets: {exc}")

# -----------------------------------------------------------------------
# Tab 2: Trade log
# -----------------------------------------------------------------------

with tab_trades:
    st.subheader("Trade & Skip Log")

    if decisions_df.empty:
        st.info("No decisions recorded yet.")
    else:
        show_skips = st.checkbox("Include skipped trades", value=False)

        display_df = decisions_df.copy()
        if not show_skips:
            display_df = display_df[~display_df.get("skipped", pd.Series(True, index=display_df.index))]

        cols_to_show = [c for c in [
            "timestamp", "market_id", "action", "calibrated_prob",
            "edge_after_costs", "confidence", "position_size",
            "reason", "skipped", "skip_reason",
        ] if c in display_df.columns]

        st.dataframe(
            display_df[cols_to_show].sort_values("timestamp", ascending=False).head(200),
            use_container_width=True,
            height=400,
        )

        st.subheader("Skip Reason Distribution")
        if "skip_reason" in decisions_df.columns:
            skip_counts = (
                decisions_df[decisions_df.get("skipped", False) == True]["skip_reason"]
                .value_counts()
                .head(10)
            )
            if not skip_counts.empty:
                st.bar_chart(skip_counts)

# -----------------------------------------------------------------------
# Tab 3: Bankroll & Drawdown
# -----------------------------------------------------------------------

with tab_bankroll:
    st.subheader("Bankroll Curve")

    # Load from in-memory risk manager state if available
    try:
        from polymarket_alpha.risk.manager import RiskManager
        risk = RiskManager()
        state = risk.get_state()
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Current Bankroll", f"${state.bankroll:,.2f}")
        bcol2.metric("Daily PnL", f"${state.daily_pnl:,.2f}")
        bcol3.metric("Drawdown", f"{state.current_drawdown_pct:.2%}")

        if state.halted:
            st.error(f"⚠️ SYSTEM HALTED: {state.halt_reason}")

        if state.category_exposure:
            st.subheader("Category Exposure")
            exp_df = pd.DataFrame(
                list(state.category_exposure.items()),
                columns=["Category", "Exposure ($)"],
            ).sort_values("Exposure ($)", ascending=False)
            st.bar_chart(exp_df.set_index("Category"))

    except Exception as exc:
        st.warning(f"Risk state unavailable: {exc}")

    # If we have decisions with equity data, plot it
    if not decisions_df.empty and "timestamp" in decisions_df.columns:
        st.subheader("Decision Timeline")
        timeline = decisions_df[["timestamp", "edge_after_costs", "confidence"]].dropna()
        if not timeline.empty:
            st.line_chart(timeline.set_index("timestamp")[["edge_after_costs", "confidence"]])

# -----------------------------------------------------------------------
# Tab 4: Calibration
# -----------------------------------------------------------------------

with tab_calibration:
    st.subheader("Calibration Metrics")

    if cal_df.empty:
        st.info("No calibration metrics recorded yet.")
    else:
        # Brier score over time
        if "brier_score" in cal_df.columns and "timestamp" in cal_df.columns:
            st.subheader("Brier Score Over Time (lower is better)")
            chart_df = cal_df.set_index("timestamp")[["brier_score", "log_loss"]].sort_index()
            st.line_chart(chart_df)

        # ECE and overconfidence
        if "ece" in cal_df.columns:
            recent = cal_df.iloc[-1]
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Brier Score", f"{recent['brier_score']:.4f}")
            cc2.metric("Log Loss", f"{recent['log_loss']:.4f}")
            cc3.metric("ECE", f"{recent['ece']:.4f}")
            cc4.metric("Overconfidence", f"{recent['overconfidence']:.4f}")

        # Drift alert
        if len(cal_df) >= 2 and "brier_score" in cal_df.columns:
            try:
                from polymarket_alpha.config import get_config
                cfg = get_config()
                baseline = cal_df["brier_score"].iloc[0]
                recent_bs = cal_df["brier_score"].iloc[-1]
                drift = recent_bs - baseline
                if drift > cfg.calibration_drift_threshold:
                    st.warning(
                        f"⚠️ Calibration drift detected: Brier score increased by "
                        f"{drift:.4f} (threshold: {cfg.calibration_drift_threshold})"
                    )
                else:
                    st.success("✅ Calibration within acceptable range")
            except Exception:
                pass

        st.dataframe(cal_df.tail(50), use_container_width=True)

# -----------------------------------------------------------------------
# Tab 5: Signal breakdown
# -----------------------------------------------------------------------

with tab_signals:
    st.subheader("Signal Breakdown by Market")

    if decisions_df.empty:
        st.info("No decisions to show.")
    else:
        if "signal_probs" in decisions_df.columns:
            # Parse signal_probs column
            def parse_sp(val: Any) -> Dict:
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return {}
                return {}

            sig_expanded = decisions_df["signal_probs"].apply(parse_sp).apply(pd.Series)
            sig_df = pd.concat(
                [decisions_df[["timestamp", "market_id", "calibrated_prob", "action"]].reset_index(drop=True),
                 sig_expanded.reset_index(drop=True)],
                axis=1,
            )
            st.dataframe(sig_df.tail(100), use_container_width=True)

            # Correlation between signals
            signal_cols = [c for c in sig_expanded.columns if c in ["llm", "microstructure", "heuristic", "ensemble"]]
            if len(signal_cols) >= 2:
                st.subheader("Signal Distributions")
                st.bar_chart(sig_expanded[signal_cols].mean().rename("Mean Probability by Signal"))

        st.subheader("Category-Level Performance")
        if "action" in decisions_df.columns and "category" in decisions_df.columns:
            cat_perf = (
                decisions_df[decisions_df["skipped"] != True]
                .groupby("category")[["edge_after_costs", "confidence", "position_size"]]
                .mean()
                .round(4)
            )
            if not cat_perf.empty:
                st.dataframe(cat_perf, use_container_width=True)

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------

st.divider()
st.caption(
    "⚠️ This system provides no guarantees of profitability. "
    "All trading involves risk. Past performance does not predict future results. "
    "Run in paper trading mode before considering live trading."
)
