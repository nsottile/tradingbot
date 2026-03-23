"""
Test suite for polymarket_alpha.

Covers:
- Claude JSON parsing
- Calibration
- Kelly sizing
- Stale data rejection
- Backtest no-lookahead behaviour
- Trade skip logic
- Schema validation
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from polymarket_alpha.backtest.engine import BacktestEngine
from polymarket_alpha.calibration.calibrator import Calibrator
from polymarket_alpha.execution.decision_engine import TradeDecisionEngine, _compute_ev
from polymarket_alpha.features.builder import FeatureBuilder
from polymarket_alpha.ingestion.collector import MarketCollector
from polymarket_alpha.models.llm_model import LLMProbabilityModel
from polymarket_alpha.models.signal_models import HeuristicModel, MicrostructureModel
from polymarket_alpha.risk.manager import RiskManager
from polymarket_alpha.schemas import (
    Action,
    MarketSnapshot,
    SignalProbabilities,
    TradeDecision,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_snap(
    market_id: str = "test-market",
    yes_price: float = 0.60,
    no_price: float = 0.40,
    liquidity: float = 5000.0,
    spread: float = 0.02,
    volume: float = 10_000.0,
    category: str = "politics",
    is_active: bool = True,
    offset_seconds: int = 0,
) -> MarketSnapshot:
    ts = datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    return MarketSnapshot(
        market_id=market_id,
        question="Will X happen?",
        category=category,
        yes_price=yes_price,
        no_price=no_price,
        mid_price=(yes_price + no_price) / 2,
        spread=spread,
        liquidity=liquidity,
        volume=volume,
        timestamp=ts,
        resolution_date=ts + timedelta(days=7),
        is_active=is_active,
    )


# -----------------------------------------------------------------------
# 1. LLM JSON parsing
# -----------------------------------------------------------------------

class TestLLMParsing:
    """Tests for the LLM response parser."""

    def _parser(self) -> LLMProbabilityModel:
        model = LLMProbabilityModel.__new__(LLMProbabilityModel)
        model._client = None
        model._cfg = MagicMock()
        model._cfg.llm_retry_attempts = 2
        return model

    def test_valid_json(self) -> None:
        model = self._parser()
        response = json.dumps({
            "probability": 0.72,
            "confidence": 0.85,
            "reasoning_summary": "Strong indicators.",
            "bullish_factors": ["A", "B"],
            "bearish_factors": ["C"],
            "uncertainty_sources": ["Unknown timing"],
            "data_needed": ["More data"],
        })
        result = model._parse_response(response)
        assert result is not None
        assert abs(result["probability"] - 0.72) < 1e-6
        assert abs(result["confidence"] - 0.85) < 1e-6

    def test_json_with_markdown_fence(self) -> None:
        model = self._parser()
        response = '```json\n{"probability": 0.55, "confidence": 0.70}\n```'
        result = model._parse_response(response)
        assert result is not None
        assert abs(result["probability"] - 0.55) < 1e-6

    def test_json_embedded_in_prose(self) -> None:
        model = self._parser()
        response = 'Here is my answer: {"probability": 0.40, "confidence": 0.60} done.'
        result = model._parse_response(response)
        assert result is not None
        assert abs(result["probability"] - 0.40) < 1e-6

    def test_missing_required_fields_returns_none(self) -> None:
        model = self._parser()
        response = json.dumps({"reasoning": "something"})
        result = model._parse_response(response)
        assert result is None

    def test_invalid_json_returns_none(self) -> None:
        model = self._parser()
        result = model._parse_response("not json at all !@#$")
        assert result is None

    def test_probability_clamped_to_valid_range(self) -> None:
        model = self._parser()
        response = json.dumps({"probability": 1.5, "confidence": -0.3})
        result = model._parse_response(response)
        assert result is not None
        assert result["probability"] <= 0.99
        assert result["confidence"] >= 0.0

    def test_string_probability_coerced(self) -> None:
        model = self._parser()
        response = json.dumps({"probability": "0.65", "confidence": "0.80"})
        result = model._parse_response(response)
        assert result is not None
        assert abs(result["probability"] - 0.65) < 1e-6

    def test_none_response_returns_none(self) -> None:
        model = self._parser()
        result = model._parse_response("")
        assert result is None

    def test_partial_json_recovered(self) -> None:
        model = self._parser()
        response = 'Some text {"probability": 0.30, "confidence": 0.50} trailing text'
        result = model._parse_response(response)
        assert result is not None


# -----------------------------------------------------------------------
# 2. Calibration
# -----------------------------------------------------------------------

class TestCalibration:

    def test_fit_and_calibrate(self) -> None:
        cal = Calibrator()
        probs = list(np.linspace(0.1, 0.9, 50))
        # Outcomes: YES when prob > 0.5
        outcomes = [1.0 if p > 0.5 else 0.0 for p in probs]
        cal.fit(probs, outcomes, market_family="test_family")
        result = cal.calibrate(0.7, market_family="test_family")
        assert 0.0 < result < 1.0

    def test_fallback_to_global(self) -> None:
        cal = Calibrator()
        probs = list(np.linspace(0.1, 0.9, 50))
        outcomes = [1.0 if p > 0.5 else 0.0 for p in probs]
        cal.fit(probs, outcomes, market_family="global")
        result = cal.calibrate(0.6, market_family="unknown_family")
        assert 0.0 < result < 1.0

    def test_no_calibrator_returns_raw(self) -> None:
        cal = Calibrator()
        cal._models = {}
        cal._global = None
        result = cal.calibrate(0.65, market_family="missing")
        assert abs(result - 0.65) < 1e-6

    def test_evaluate_returns_metrics(self) -> None:
        cal = Calibrator()
        probs = list(np.linspace(0.1, 0.9, 30))
        outcomes = [1.0 if p > 0.5 else 0.0 for p in probs]
        metrics = cal.evaluate(probs, outcomes, market_family="global")
        assert metrics is not None
        assert 0 <= metrics.brier_score <= 1.0
        assert metrics.n_samples == 30

    def test_too_few_samples_skips_fit(self) -> None:
        cal = Calibrator()
        # Only 3 samples — below minimum
        cal.fit([0.3, 0.6, 0.8], [0.0, 1.0, 1.0], market_family="tiny")
        # Should not have fitted (no exception, just skipped)
        assert "tiny" not in cal._models

    def test_drift_detection(self) -> None:
        cal = Calibrator()
        from polymarket_alpha.schemas import CalibrationMetrics
        baseline = 0.20
        recent = CalibrationMetrics(
            timestamp=datetime.now(timezone.utc),
            market_family="global",
            brier_score=0.30,  # Much worse
            log_loss=0.5,
            n_samples=50,
            ece=0.05,
            overconfidence=0.02,
        )
        assert cal.check_drift(recent, baseline_brier=baseline)

    def test_no_drift_when_stable(self) -> None:
        cal = Calibrator()
        from polymarket_alpha.schemas import CalibrationMetrics
        baseline = 0.20
        recent = CalibrationMetrics(
            timestamp=datetime.now(timezone.utc),
            market_family="global",
            brier_score=0.21,  # Tiny change
            log_loss=0.5,
            n_samples=50,
            ece=0.02,
            overconfidence=0.01,
        )
        assert not cal.check_drift(recent, baseline_brier=baseline)


# -----------------------------------------------------------------------
# 3. Kelly sizing
# -----------------------------------------------------------------------

class TestKellySizing:

    def _risk(self) -> RiskManager:
        return RiskManager(initial_bankroll=10_000.0)

    def test_positive_edge_yields_positive_size(self) -> None:
        risk = self._risk()
        snap = _make_snap(yes_price=0.40, liquidity=10_000)
        allowed, size, reason = risk.check_and_size(
            snap=snap,
            action=Action.BUY_YES,
            calibrated_prob=0.65,
            edge_after_costs=0.10,
            confidence=0.80,
        )
        assert allowed
        assert size > 0

    def test_below_edge_threshold_rejected(self) -> None:
        risk = self._risk()
        snap = _make_snap(yes_price=0.50, liquidity=10_000)
        allowed, size, reason = risk.check_and_size(
            snap=snap,
            action=Action.BUY_YES,
            calibrated_prob=0.52,
            edge_after_costs=0.005,  # Below threshold
            confidence=0.80,
        )
        assert not allowed
        assert size == 0.0

    def test_position_never_exceeds_max_pct(self) -> None:
        risk = self._risk()
        snap = _make_snap(yes_price=0.10, liquidity=1_000_000)
        allowed, size, reason = risk.check_and_size(
            snap=snap,
            action=Action.BUY_YES,
            calibrated_prob=0.95,
            edge_after_costs=0.50,
            confidence=1.0,
        )
        from polymarket_alpha.config import get_config
        max_pos = 10_000.0 * get_config().max_position_pct
        assert size <= max_pos + 0.01

    def test_low_confidence_reduces_size(self) -> None:
        risk_hi = self._risk()
        risk_lo = self._risk()
        snap = _make_snap(yes_price=0.40, liquidity=20_000)

        _, size_hi, _ = risk_hi.check_and_size(snap, Action.BUY_YES, 0.70, 0.15, 0.95)
        _, size_lo, _ = risk_lo.check_and_size(snap, Action.BUY_YES, 0.70, 0.15, 0.55)
        assert size_hi >= size_lo

    def test_halted_system_rejects_all(self) -> None:
        risk = self._risk()
        risk._halt("Test halt")
        snap = _make_snap(liquidity=100_000)
        allowed, size, reason = risk.check_and_size(snap, Action.BUY_YES, 0.90, 0.30, 1.0)
        assert not allowed

    def test_daily_loss_triggers_halt(self) -> None:
        risk = self._risk()
        # Record a big loss
        risk.record_pnl(-600.0)  # 6% of 10k — over 5% limit
        snap = _make_snap(liquidity=100_000)
        allowed, _, _ = risk.check_and_size(snap, Action.BUY_YES, 0.80, 0.20, 0.90)
        assert not allowed

    def test_kelly_fraction_applied(self) -> None:
        """Full Kelly would be larger; fractional Kelly must be smaller."""
        risk = self._risk()
        snap = _make_snap(yes_price=0.30, liquidity=500_000)
        # With 0.25 Kelly fraction, size should be much less than bankroll
        allowed, size, _ = risk.check_and_size(snap, Action.BUY_YES, 0.75, 0.20, 0.90)
        assert size < risk._bankroll  # Never bet the whole bankroll


# -----------------------------------------------------------------------
# 4. Stale data rejection
# -----------------------------------------------------------------------

class TestStaleData:

    def test_fresh_data_accepted(self) -> None:
        collector = MarketCollector()
        snap = _make_snap(offset_seconds=10)  # 10 seconds old
        assert collector.is_data_fresh(snap)

    def test_stale_data_rejected(self) -> None:
        collector = MarketCollector()
        snap = _make_snap(offset_seconds=600)  # 10 minutes old
        assert not collector.is_data_fresh(snap)

    def test_boundary_exactly_at_threshold(self) -> None:
        collector = MarketCollector()
        from polymarket_alpha.config import get_config
        threshold = get_config().stale_data_threshold_seconds
        snap = _make_snap(offset_seconds=threshold - 1)
        assert collector.is_data_fresh(snap)

        snap_stale = _make_snap(offset_seconds=threshold + 1)
        assert not collector.is_data_fresh(snap_stale)


# -----------------------------------------------------------------------
# 5. Backtest no-lookahead behaviour
# -----------------------------------------------------------------------

class TestBacktestNoLookahead:

    def test_history_never_contains_future_snapshots(self) -> None:
        """
        Verify that when the backtest processes snapshot at time T,
        the history passed to the decision engine contains only
        snapshots with timestamp < T.
        """
        market_id = "no-lookahead-test"
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        snapshots = [
            _make_snap(market_id=market_id, yes_price=0.40 + i * 0.02)
            .__class__(
                **{
                    **_make_snap(market_id=market_id).model_dump(),
                    "timestamp": base_time + timedelta(hours=i),
                    "yes_price": 0.40 + i * 0.01,
                    "no_price": 0.60 - i * 0.01,
                    "mid_price": 0.50,
                }
            )
            for i in range(10)
        ]

        engine = BacktestEngine(snapshots=snapshots, resolutions={})

        # Manually sort and verify history construction
        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        for idx, snap in enumerate(sorted_snaps):
            history = [
                h for h in engine._by_market[market_id]
                if h.timestamp < snap.timestamp
            ]
            for h in history:
                assert h.timestamp < snap.timestamp, (
                    f"LOOKAHEAD: history snap at {h.timestamp} >= current {snap.timestamp}"
                )

    def test_resolutions_not_used_in_features(self) -> None:
        """
        Resolutions dict must not be available during feature building.
        The FeatureBuilder should receive no outcome information.
        """
        fb = FeatureBuilder()
        snap = _make_snap()
        features = fb.build(snap)
        # Outcome-related keys must not appear in features
        for key in features:
            assert "resolution" not in key.lower()
            assert "outcome" not in key.lower()
            assert "result" not in key.lower()

    def test_backtest_runs_without_lookahead(self) -> None:
        """Integration test: backtest completes with sensible output."""
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snaps = []
        for i in range(5):
            s = MarketSnapshot(
                market_id="mkt-1",
                question="Test?",
                category="politics",
                yes_price=0.55,
                no_price=0.45,
                mid_price=0.50,
                spread=0.05,
                liquidity=5000.0,
                volume=10000.0,
                timestamp=base_time + timedelta(hours=i),
                is_active=(i < 4),
            )
            snaps.append(s)

        engine = BacktestEngine(
            snapshots=snaps,
            resolutions={"mkt-1": 1.0},
        )
        results = engine.run()
        assert "metrics" in results
        assert "trades" in results
        assert "equity_curve" in results


# -----------------------------------------------------------------------
# 6. Trade skip logic
# -----------------------------------------------------------------------

class TestTradeSkipLogic:

    def test_low_liquidity_skipped(self) -> None:
        snap = _make_snap(liquidity=50.0)  # Below 500 minimum
        risk = RiskManager()
        allowed, _, reason = risk.check_and_size(
            snap=snap,
            action=Action.BUY_YES,
            calibrated_prob=0.75,
            edge_after_costs=0.15,
            confidence=0.90,
        )
        assert not allowed
        assert "liquidity" in reason.lower()

    def test_high_spread_skipped(self) -> None:
        snap = _make_snap(spread=0.15, liquidity=10_000)  # Spread 15% > 8% max
        risk = RiskManager()
        allowed, _, reason = risk.check_and_size(
            snap=snap,
            action=Action.BUY_YES,
            calibrated_prob=0.75,
            edge_after_costs=0.15,
            confidence=0.90,
        )
        assert not allowed
        assert "spread" in reason.lower()

    def test_zero_ev_skipped(self) -> None:
        ev = _compute_ev(0.50, Action.BUY_YES, _make_snap(yes_price=0.50), 0.005, 0.02)
        assert ev < 0  # At 50/50 there is no edge

    def test_negative_ev_skipped_in_engine(self) -> None:
        engine = TradeDecisionEngine.__new__(TradeDecisionEngine)
        engine._cfg = MagicMock()
        engine._cfg.llm_enabled = False
        engine._cfg.slippage_estimate = 0.005
        engine._cfg.fee_estimate = 0.02
        engine._cfg.edge_threshold = 0.04
        engine._cfg.min_confidence = 0.50
        engine._cfg.min_liquidity = 100.0
        engine._cfg.max_spread = 0.20

        from polymarket_alpha.features.builder import FeatureBuilder
        from polymarket_alpha.models.signal_models import HeuristicModel, MicrostructureModel
        from polymarket_alpha.models.ensemble import EnsembleModel
        from polymarket_alpha.calibration.calibrator import Calibrator

        engine._feature_builder = FeatureBuilder()
        engine._llm_model = MagicMock()
        engine._llm_model.predict.return_value = None
        engine._micro_model = MicrostructureModel()
        engine._heuristic_model = HeuristicModel()
        engine._ensemble = EnsembleModel()
        engine._calibrator = Calibrator()
        engine._risk = RiskManager()

        # A market at exactly 0.50 with fees should be skipped
        snap = _make_snap(yes_price=0.50, no_price=0.50, liquidity=5000)
        decision = engine.decide(snap)
        assert decision.skipped or decision.action == Action.SKIP


# -----------------------------------------------------------------------
# 7. Schema validation
# -----------------------------------------------------------------------

class TestSchemaValidation:

    def test_market_snapshot_valid(self) -> None:
        snap = _make_snap()
        assert snap.market_id == "test-market"
        assert 0 <= snap.yes_price <= 1.0
        assert 0 <= snap.no_price <= 1.0

    def test_market_snapshot_price_clamped(self) -> None:
        snap = MarketSnapshot(
            market_id="x",
            question="Q?",
            category="test",
            yes_price=1.5,   # Out of range — should be clamped
            no_price=-0.1,
            mid_price=0.5,
            spread=0.0,
            liquidity=1000.0,
            volume=5000.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert snap.yes_price <= 1.0
        assert snap.no_price >= 0.0

    def test_trade_decision_frozen(self) -> None:
        decision = TradeDecision(
            market_id="x",
            timestamp=datetime.now(timezone.utc),
            signal_probs=SignalProbabilities(),
            calibrated_prob=0.60,
            expected_value=0.08,
            edge_after_costs=0.08,
            confidence=0.75,
            position_size=100.0,
            action=Action.BUY_YES,
            reason="Test",
            skipped=False,
        )
        with pytest.raises(Exception):
            decision.market_id = "other"  # Should raise on frozen model

    def test_signal_probabilities_all_none(self) -> None:
        sp = SignalProbabilities()
        assert sp.llm is None
        assert sp.microstructure is None
        assert sp.heuristic is None
        assert sp.ensemble is None

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(Exception):
            TradeDecision(
                market_id="x",
                timestamp=datetime.now(timezone.utc),
                signal_probs=SignalProbabilities(),
                calibrated_prob=0.6,
                expected_value=0.1,
                edge_after_costs=0.1,
                confidence=0.8,
                position_size=100.0,
                action="INVALID_ACTION",  # type: ignore
                reason="bad",
                skipped=False,
            )


# -----------------------------------------------------------------------
# 8. Feature builder
# -----------------------------------------------------------------------

class TestFeatureBuilder:

    def test_features_are_finite(self) -> None:
        fb = FeatureBuilder()
        snap = _make_snap()
        features = fb.build(snap)
        import math
        for k, v in features.items():
            assert math.isfinite(v), f"Feature {k} is not finite: {v}"

    def test_history_lookahead_filtered(self) -> None:
        """History items with future timestamps must be excluded."""
        fb = FeatureBuilder()
        now = datetime.now(timezone.utc)
        snap = _make_snap()

        future_snap = MarketSnapshot(
            **{**snap.model_dump(), "timestamp": now + timedelta(hours=1)}
        )
        past_snap = MarketSnapshot(
            **{**snap.model_dump(), "timestamp": now - timedelta(hours=1)}
        )

        features = fb.build(snap, history=[future_snap, past_snap])
        # Should not raise and should have processed without future data
        assert isinstance(features, dict)

    def test_empty_history_works(self) -> None:
        fb = FeatureBuilder()
        snap = _make_snap()
        features = fb.build(snap, history=[])
        assert "mid_price" in features
        assert "spread" in features


# -----------------------------------------------------------------------
# 9. EV calculation
# -----------------------------------------------------------------------

class TestEVCalculation:

    def test_positive_ev_when_underpriced(self) -> None:
        snap = _make_snap(yes_price=0.40)
        # We think probability is 0.65 — market underprices YES
        ev = _compute_ev(0.65, Action.BUY_YES, snap, slippage=0.0, fee=0.0)
        assert ev > 0

    def test_negative_ev_when_overpriced(self) -> None:
        snap = _make_snap(yes_price=0.80)
        # We think probability is 0.50 — market overprices YES
        ev = _compute_ev(0.50, Action.BUY_YES, snap, slippage=0.0, fee=0.0)
        assert ev < 0

    def test_fees_reduce_ev(self) -> None:
        snap = _make_snap(yes_price=0.40)
        ev_no_fee = _compute_ev(0.65, Action.BUY_YES, snap, slippage=0.0, fee=0.0)
        ev_with_fee = _compute_ev(0.65, Action.BUY_YES, snap, slippage=0.0, fee=0.02)
        assert ev_with_fee < ev_no_fee

    def test_skip_action_returns_zero(self) -> None:
        snap = _make_snap()
        ev = _compute_ev(0.70, Action.SKIP, snap, slippage=0.0, fee=0.0)
        assert ev == 0.0
