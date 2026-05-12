"""Tests for evaluation metrics module."""

import pytest
import numpy as np
from src.evaluation.metrics import EvaluationMetrics


@pytest.fixture
def predictions():
    """Create sample predictions."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.7, 0.3],
        ]
    )
    return y_true, y_pred, y_proba


def test_calculate_metrics(predictions):
    """Test metric calculation."""
    y_true, y_pred, _ = predictions
    evaluator = EvaluationMetrics(y_true, y_pred)
    metrics = evaluator.calculate_metrics()

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_calculate_auc(predictions):
    """Test AUC calculation."""
    y_true, _, y_proba = predictions
    evaluator = EvaluationMetrics(y_true, y_proba[:, 1], is_proba=True)
    auc = evaluator.calculate_auc()

    assert 0 <= auc <= 1


def test_confusion_matrix(predictions):
    """Test confusion matrix."""
    y_true, y_pred, _ = predictions
    evaluator = EvaluationMetrics(y_true, y_pred)
    cm = evaluator.get_confusion_matrix()

    assert cm.shape == (2, 2)
