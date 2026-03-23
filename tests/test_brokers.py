"""Broker and execution router tests."""

from __future__ import annotations

from datetime import datetime, timezone

from polymarket_alpha.brokers.paper import PaperBroker
from polymarket_alpha.brokers.types import BrokerVenue, OrderRequest, OrderSide
from polymarket_alpha.data.schemas import AssetClass, NormalizedMarketSnapshot
from polymarket_alpha.engine.control import AIControlState
from polymarket_alpha.engine.ledger import PortfolioLedger
from polymarket_alpha.engine.router import ExecutionRouter


def _snap() -> NormalizedMarketSnapshot:
    return NormalizedMarketSnapshot(
        market_id="m1",
        symbol="m1",
        source="polymarket",
        asset_class=AssetClass.PREDICTION,
        venue="polymarket",
        name="Test?",
        price=0.5,
        timestamp=datetime.now(timezone.utc),
        raw_payload={"_yes_token_id": "t1", "_no_token_id": "t2", "_yes_price": 0.5, "_no_price": 0.5},
    )


def test_paper_broker_buy_updates_ledger() -> None:
    ledger = PortfolioLedger(initial_capital=10_000.0)
    b = PaperBroker(ledger)
    req = OrderRequest(
        venue=BrokerVenue.PAPER,
        market_id="m1",
        symbol="m1",
        side=OrderSide.BUY,
        notional_usd=100.0,
        limit_price=0.5,
    )
    r = b.place_order(req)
    assert r.success
    assert "m1" in ledger.positions


def test_router_uses_paper_when_simulation_mode() -> None:
    ledger = PortfolioLedger(initial_capital=10_000.0)
    ctrl = AIControlState(simulation_mode=True, live_trading_confirmed=True)
    router = ExecutionRouter(ledger, ctrl)
    assert router.uses_paper_path() is True


def test_router_uses_paper_when_kill_switch() -> None:
    ledger = PortfolioLedger(initial_capital=10_000.0)
    ctrl = AIControlState(simulation_mode=False, live_trading_confirmed=True, kill_switch=True)
    router = ExecutionRouter(ledger, ctrl)
    assert router.uses_paper_path() is True


def test_router_route_buy_prediction_paper() -> None:
    ledger = PortfolioLedger(initial_capital=10_000.0)
    ctrl = AIControlState(simulation_mode=True)
    router = ExecutionRouter(ledger, ctrl)
    res = router.route_buy_prediction(_snap(), 200.0, 0.5)
    assert res.success
    assert "m1" in ledger.positions
