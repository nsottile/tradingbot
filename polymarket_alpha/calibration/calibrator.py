"""
Calibrator: post-hoc probability calibration.

Trains on historical out-of-sample data only (no lookahead).
Uses isotonic regression. Falls back to the global calibrator
when a market-family calibrator has insufficient data.

Includes:
- Brier score
- Log loss
- Expected Calibration Error (ECE)
- Calibration drift monitoring
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from polymarket_alpha.config import get_config
from polymarket_alpha.schemas import CalibrationMetrics
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MODEL_DIR = Path("models/calibrators")


class Calibrator:
    """
    Isotonic regression calibrator with per-family support.

    Usage
    -----
    cal = Calibrator()
    cal.fit(probs, outcomes, market_family="politics")
    calibrated = cal.calibrate(raw_prob, market_family="politics")
    metrics = cal.evaluate(probs, outcomes, market_family="politics")
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._models: Dict[str, IsotonicRegression] = {}
        self._global: Optional[IsotonicRegression] = None
        self._load_all()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        probs: List[float],
        outcomes: List[float],
        market_family: str = "global",
    ) -> "Calibrator":
        """
        Fit isotonic regression calibrator.

        Parameters
        ----------
        probs:
            Raw model probabilities.
        outcomes:
            Binary outcomes (1.0 = YES resolved, 0.0 = NO resolved).
        market_family:
            "global" or a specific market family string.
        """
        if len(probs) < self._cfg.calibration_min_samples:
            logger.warning(
                "Too few samples to fit calibrator",
                extra={"family": market_family, "n": len(probs)},
            )
            return self

        X = np.array(probs, dtype=float)
        y = np.array(outcomes, dtype=float)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X, y)

        if market_family == "global":
            self._global = iso
        else:
            self._models[market_family] = iso

        self._save(market_family, iso)
        logger.info(
            "Calibrator fitted",
            extra={"family": market_family, "n_samples": len(probs)},
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(
        self,
        raw_prob: float,
        market_family: str = "global",
    ) -> float:
        """
        Return calibrated probability.

        Falls back to global calibrator if no family-specific one exists.
        Falls back to the raw probability if no calibrator is available.
        """
        model = self._models.get(market_family) or self._global
        if model is None:
            # No calibrator available — return raw probability
            return max(0.01, min(0.99, raw_prob))

        try:
            calibrated = float(model.predict([raw_prob])[0])
            return max(0.01, min(0.99, calibrated))
        except Exception as exc:
            logger.warning(
                "Calibration inference failed",
                extra={"error": str(exc), "raw_prob": raw_prob},
            )
            return max(0.01, min(0.99, raw_prob))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        probs: List[float],
        outcomes: List[float],
        market_family: str = "global",
    ) -> Optional[CalibrationMetrics]:
        """Compute Brier score, log loss, and ECE."""
        if len(probs) < 5:
            return None

        X = np.array(probs, dtype=float)
        y = np.array(outcomes, dtype=float)

        brier = brier_score_loss(y, X)
        ll = log_loss(y, X)
        ece = self._compute_ece(X, y)
        overconf = float(np.mean(X) - np.mean(y))

        return CalibrationMetrics(
            timestamp=datetime.now(timezone.utc),
            market_family=market_family,
            brier_score=brier,
            log_loss=ll,
            n_samples=len(probs),
            ece=ece,
            overconfidence=overconf,
        )

    def check_drift(
        self,
        recent_metrics: CalibrationMetrics,
        baseline_brier: float,
    ) -> bool:
        """
        Return True if calibration has drifted beyond threshold.
        Drift is defined as Brier score increase above baseline.
        """
        delta = recent_metrics.brier_score - baseline_brier
        return delta > self._cfg.calibration_drift_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_ece(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(outcomes[mask]))
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
        return ece

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, family: str, model: IsotonicRegression) -> None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = _MODEL_DIR / f"{family}.pkl"
        with path.open("wb") as f:
            pickle.dump(model, f)

    def _load_all(self) -> None:
        if not _MODEL_DIR.exists():
            return
        for path in _MODEL_DIR.glob("*.pkl"):
            family = path.stem
            try:
                with path.open("rb") as f:
                    model = pickle.load(f)
                if family == "global":
                    self._global = model
                else:
                    self._models[family] = model
                logger.info("Loaded calibrator", extra={"family": family})
            except Exception as exc:
                logger.warning(
                    "Failed to load calibrator",
                    extra={"family": family, "error": str(exc)},
                )

    def get_reliability_curve(
        self,
        probs: List[float],
        outcomes: List[float],
        n_bins: int = 10,
    ) -> Tuple[List[float], List[float]]:
        """Return (mean_predicted, fraction_positive) for reliability diagram."""
        X = np.array(probs, dtype=float)
        y = np.array(outcomes, dtype=float)
        frac_pos, mean_pred = calibration_curve(y, X, n_bins=n_bins, strategy="uniform")
        return list(mean_pred), list(frac_pos)
