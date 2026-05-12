"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict


class EvaluationMetrics:
    """Calculate evaluation metrics for classification."""

    def __init__(
        self, y_true: np.ndarray, y_pred_or_proba: np.ndarray, is_proba: bool = False
    ) -> None:
        """
        Initialize evaluator.

        Args:
            y_true: True labels
            y_pred_or_proba: Predictions or probabilities
            is_proba: Whether y_pred_or_proba are probabilities
        """
        self.y_true = y_true

        if is_proba:
            self.y_pred = (y_pred_or_proba > 0.5).astype(int)
            self.y_proba = y_pred_or_proba
        else:
            self.y_pred = y_pred_or_proba
            self.y_proba = None

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1": f1_score(self.y_true, self.y_pred),
        }

        if self.y_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(self.y_true, self.y_proba)
            except Exception:
                metrics["auc"] = 0.0

        return metrics

    def calculate_auc(self) -> float:
        """
        Calculate ROC AUC.

        Returns:
            AUC score
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for AUC")

        return roc_auc_score(self.y_true, self.y_proba)

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self) -> str:
        """
        Get detailed classification report.

        Returns:
            Formatted report string
        """
        return classification_report(self.y_true, self.y_pred)
