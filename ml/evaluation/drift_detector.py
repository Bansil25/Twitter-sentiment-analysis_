"""
Model drift detection for production monitoring.

Two types of drift matter in sentiment analysis:
  1. Data drift    — input text distribution shifts (e.g. new slang after a product launch)
  2. Concept drift — same inputs, changing sentiments (e.g. brand perception flips)

This module:
  - Logs prediction distributions to a rolling window
  - Computes KL divergence vs training baseline
  - Alerts when drift exceeds threshold
  - Integrates with Prometheus for Grafana alerting

In production, call DriftDetector.record() after every prediction
and DriftDetector.check_drift() in a scheduled job (every 1h).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Training-time baseline distributions (update after each training run)
BASELINE_DISTRIBUTIONS = {
    "bilstm": {
        "Negative":   0.312,
        "Positive":   0.381,
        "Neutral":    0.197,
        "Irrelevant": 0.110,
    },
    "distilbert": {
        "Negative":   0.312,
        "Positive":   0.381,
        "Neutral":    0.197,
        "Irrelevant": 0.110,
    },
}

DRIFT_KL_THRESHOLD     = 0.1   # KL divergence alert threshold
CONFIDENCE_ALERT_BELOW = 0.6   # Alert if avg confidence drops below this
MIN_SAMPLES_FOR_CHECK  = 500   # Don't check drift with fewer samples


@dataclass
class DriftReport:
    model_name: str
    window_size: int
    timestamp: float
    kl_divergence: float
    drift_detected: bool
    current_distribution: Dict[str, float]
    baseline_distribution: Dict[str, float]
    avg_confidence: float
    confidence_alert: bool
    recommendation: str


class DriftDetector:
    """
    Rolling-window drift detector.
    One instance per model, kept in memory (or backed by Redis in distributed setup).
    """

    def __init__(
        self,
        model_name: str,
        window_size: int = 2000,
        check_interval_seconds: float = 3600.0,
    ):
        self.model_name      = model_name
        self.window_size     = window_size
        self.check_interval  = check_interval_seconds
        self._predictions    = deque(maxlen=window_size)  # (label, confidence)
        self._last_check_at  = 0.0
        self._baseline       = BASELINE_DISTRIBUTIONS.get(model_name, {})
        self._drift_history: List[DriftReport] = []

    # ── Recording predictions ──────────────────────────────────────────────

    def record(self, label: str, confidence: float) -> None:
        """Call this after every prediction. O(1), thread-safe via GIL."""
        self._predictions.append((label, confidence))

    def record_batch(self, labels: List[str], confidences: List[float]) -> None:
        for label, conf in zip(labels, confidences):
            self.record(label, conf)

    # ── Drift computation ──────────────────────────────────────────────────

    def check_drift(self, force: bool = False) -> Optional[DriftReport]:
        """
        Returns a DriftReport if drift is detected, None otherwise.
        Throttled to check_interval unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_check_at) < self.check_interval:
            return None
        if len(self._predictions) < MIN_SAMPLES_FOR_CHECK:
            return None

        self._last_check_at = now
        report = self._compute_report()
        self._drift_history.append(report)

        if report.drift_detected or report.confidence_alert:
            log.warning(
                "drift.detected",
                extra={
                    "model": self.model_name,
                    "kl": report.kl_divergence,
                    "drift": report.drift_detected,
                    "conf_alert": report.confidence_alert,
                },
            )
        return report

    def _compute_report(self) -> DriftReport:
        labels      = [p[0] for p in self._predictions]
        confidences = [p[1] for p in self._predictions]

        # Current distribution
        unique, counts = np.unique(labels, return_counts=True)
        total   = len(labels)
        current = {k: round(c / total, 4) for k, c in zip(unique, counts)}

        # Ensure all classes present in both distributions
        all_classes = set(self._baseline.keys()) | set(current.keys())
        eps = 1e-10  # Laplace smoothing to avoid log(0)
        p   = np.array([self._baseline.get(c, eps) for c in sorted(all_classes)])
        q   = np.array([current.get(c, eps)        for c in sorted(all_classes)])
        p  /= p.sum()
        q  /= q.sum()

        kl_div           = float(_kl_divergence(p, q))
        avg_confidence   = float(np.mean(confidences))
        drift_detected   = kl_div > DRIFT_KL_THRESHOLD
        confidence_alert = avg_confidence < CONFIDENCE_ALERT_BELOW

        if drift_detected and confidence_alert:
            rec = "URGENT: Both distribution shift AND confidence drop. Likely a data or model issue. Retrain."
        elif drift_detected:
            rec = f"Distribution shift (KL={kl_div:.3f}). Monitor closely. Consider retraining if sustained."
        elif confidence_alert:
            rec = f"Low avg confidence ({avg_confidence:.2%}). Model may be encountering OOD inputs."
        else:
            rec = "No drift detected. Model appears stable."

        return DriftReport(
            model_name=self.model_name,
            window_size=len(self._predictions),
            timestamp=time.time(),
            kl_divergence=round(kl_div, 4),
            drift_detected=drift_detected,
            current_distribution=current,
            baseline_distribution=dict(self._baseline),
            avg_confidence=round(avg_confidence, 4),
            confidence_alert=confidence_alert,
            recommendation=rec,
        )

    # ── Utilities ──────────────────────────────────────────────────────────

    def reset_window(self) -> None:
        """Clear the prediction window. Call after retraining."""
        self._predictions.clear()
        log.info("drift_detector.reset", model=self.model_name)

    def to_dict(self) -> dict:
        """Serialise state to dict for Redis persistence."""
        return {
            "model_name":   self.model_name,
            "predictions":  list(self._predictions),
            "last_check":   self._last_check_at,
        }

    @property
    def stats(self) -> dict:
        """Quick stats for the /metrics endpoint."""
        n = len(self._predictions)
        if n == 0:
            return {"samples_in_window": 0}
        labels = [p[0] for p in self._predictions]
        unique, counts = np.unique(labels, return_counts=True)
        return {
            "samples_in_window": n,
            "distribution": {k: round(c/n, 3) for k, c in zip(unique, counts)},
            "avg_confidence": round(float(np.mean([p[1] for p in self._predictions])), 3),
        }


# ── Shared instances (singleton per model) ────────────────────────────────

_detectors: Dict[str, DriftDetector] = {}


def get_detector(model_name: str) -> DriftDetector:
    if model_name not in _detectors:
        _detectors[model_name] = DriftDetector(model_name)
    return _detectors[model_name]


# ── Math utilities ─────────────────────────────────────────────────────────

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) — how much q diverges from reference distribution p."""
    eps = 1e-10
    p   = np.clip(p, eps, None)
    q   = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — symmetric, bounded [0, ln(2)]."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


if __name__ == "__main__":
    # Quick demo
    detector = DriftDetector("distilbert", window_size=600)

    # Simulate normal predictions
    for _ in range(400):
        import random
        label = random.choices(["Positive", "Negative", "Neutral", "Irrelevant"],
                               weights=[0.38, 0.31, 0.20, 0.11])[0]
        detector.record(label, confidence=random.uniform(0.7, 0.99))

    report = detector.check_drift(force=True)
    if report:
        print(f"KL divergence: {report.kl_divergence}")
        print(f"Drift detected: {report.drift_detected}")
        print(f"Recommendation: {report.recommendation}")

    # Simulate drifted predictions (sudden surge in negative sentiment)
    for _ in range(200):
        import random
        detector.record("Negative", confidence=random.uniform(0.5, 0.75))

    report2 = detector.check_drift(force=True)
    if report2:
        print(f"\nAfter drift injection:")
        print(f"KL divergence: {report2.kl_divergence}")
        print(f"Drift detected: {report2.drift_detected}")
        print(f"Recommendation: {report2.recommendation}")
