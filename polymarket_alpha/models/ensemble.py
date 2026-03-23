"""
EnsembleModel: combines LLM, microstructure, and heuristic signals
into a single calibrated probability estimate.

Weights are learned from historical out-of-sample performance.
They are NOT hand-tuned based on future outcomes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from polymarket_alpha.schemas import MarketSnapshot, SignalProbabilities
from polymarket_alpha.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default weights when no trained weights are available.
# These are conservative: equal weight, favouring market price.
_DEFAULT_WEIGHTS = {
    "llm": 0.35,
    "microstructure": 0.40,
    "heuristic": 0.25,
}

_WEIGHTS_FILE = Path("models/ensemble_weights.json")


class EnsembleModel:
    """
    Weighted ensemble of signal models.

    Weights are loaded from disk if available (set by the training script).
    Falls back to default weights when no trained weights exist.

    The ensemble is confidence-weighted: signals with lower confidence
    are down-weighted automatically.
    """

    def __init__(self) -> None:
        self._weights = self._load_weights()

    def predict(
        self,
        snap: MarketSnapshot,
        llm_prob: Optional[float] = None,
        llm_conf: Optional[float] = None,
        micro_prob: Optional[float] = None,
        micro_conf: Optional[float] = None,
        heuristic_prob: Optional[float] = None,
        heuristic_conf: Optional[float] = None,
    ) -> Tuple[float, float, SignalProbabilities]:
        """
        Combine signals into a single (ensemble_probability, ensemble_confidence).

        Uses confidence-weighted averaging:
            P = sum(w_i * conf_i * p_i) / sum(w_i * conf_i)

        If no signals are available, falls back to market mid-price with
        very low confidence.

        Returns
        -------
        (ensemble_prob, ensemble_confidence, signal_probs)
        """
        signals = []

        if llm_prob is not None and llm_conf is not None:
            signals.append(("llm", llm_prob, llm_conf))

        if micro_prob is not None and micro_conf is not None:
            signals.append(("microstructure", micro_prob, micro_conf))

        if heuristic_prob is not None and heuristic_conf is not None:
            signals.append(("heuristic", heuristic_prob, heuristic_conf))

        sig_probs = SignalProbabilities(
            llm=llm_prob,
            microstructure=micro_prob,
            heuristic=heuristic_prob,
        )

        if not signals:
            logger.warning(
                "No signals available for ensemble",
                extra={"market_id": snap.market_id},
            )
            return snap.mid_price, 0.10, sig_probs

        # Confidence-weighted sum
        total_weight = 0.0
        weighted_sum = 0.0

        for name, prob, conf in signals:
            base_w = self._weights.get(name, 0.33)
            effective_w = base_w * conf
            weighted_sum += effective_w * prob
            total_weight += effective_w

        if total_weight < 1e-9:
            # All confidences were essentially zero
            ensemble_prob = snap.mid_price
            ensemble_conf = 0.05
        else:
            ensemble_prob = weighted_sum / total_weight
            # Ensemble confidence: weighted average of individual confidences
            conf_vals = [c for _, _, c in signals]
            ensemble_conf = float(np.mean(conf_vals))

        ensemble_prob = max(0.01, min(0.99, ensemble_prob))
        ensemble_conf = max(0.05, min(1.0, ensemble_conf))

        sig_probs_with_ensemble = SignalProbabilities(
            llm=llm_prob,
            microstructure=micro_prob,
            heuristic=heuristic_prob,
            ensemble=ensemble_prob,
        )

        return ensemble_prob, ensemble_conf, sig_probs_with_ensemble

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update ensemble weights and persist to disk."""
        total = sum(weights.values())
        if total <= 0:
            return
        self._weights = {k: v / total for k, v in weights.items()}
        self._save_weights()
        logger.info("Ensemble weights updated", extra={"weights": self._weights})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_weights(self) -> Dict[str, float]:
        if _WEIGHTS_FILE.exists():
            try:
                data = json.loads(_WEIGHTS_FILE.read_text())
                logger.info("Loaded ensemble weights", extra={"weights": data})
                return data
            except Exception as exc:
                logger.warning(
                    "Failed to load ensemble weights, using defaults",
                    extra={"error": str(exc)},
                )
        return dict(_DEFAULT_WEIGHTS)

    def _save_weights(self) -> None:
        _WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WEIGHTS_FILE.write_text(json.dumps(self._weights, indent=2))
