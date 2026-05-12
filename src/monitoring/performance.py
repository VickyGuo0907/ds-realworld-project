"""Performance tracking and drift detection."""

import pandas as pd
from typing import Dict


class PerformanceTracker:
    """Track performance and detect sources/prediction drift."""

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self.baseline_stats = None

    def calculate_drift(
        self, baseline: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate sources drift between baseline and current sources.

        Args:
            baseline: Baseline feature distribution
            current: Current feature distribution
            threshold: Drift threshold

        Returns:
            Dictionary of drift scores per feature
        """
        drift_scores = {}

        for col in baseline.columns:
            baseline_mean = baseline[col].mean()
            current_mean = current[col].mean()

            # Calculate percentage change
            if baseline_mean != 0:
                drift = abs(current_mean - baseline_mean) / abs(baseline_mean)
            else:
                drift = 0.0

            drift_scores[col] = drift

        return drift_scores

    def set_baseline(self, data: pd.DataFrame) -> None:
        """
        Set baseline for drift detection.

        Args:
            data: Baseline sources
        """
        self.baseline_stats = {"mean": data.mean(), "std": data.std()}
