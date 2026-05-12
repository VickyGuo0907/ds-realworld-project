import pytest
import pandas as pd
from src.monitoring.logger import PredictionLogger
from src.monitoring.performance import PerformanceTracker
import os

def test_log_prediction():
    """Test logging predictions"""
    log_file = 'test_predictions.log'
    logger = PredictionLogger(log_file)

    logger.log_prediction(
        features=[1.0, 2.0, 3.0],
        prediction=1,
        probability=0.85,
        true_label=None
    )

    # Verify log file was created
    assert os.path.exists(log_file)
    os.remove(log_file)

def test_calculate_drift():
    """Test drift calculation"""
    tracker = PerformanceTracker()

    baseline_features = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

    new_features = pd.DataFrame({
        'feature1': [1.1, 2.1, 3.1, 4.1, 5.1],
        'feature2': [11, 21, 31, 41, 51]
    })

    drift = tracker.calculate_drift(baseline_features, new_features)
    assert isinstance(drift, dict)
    assert 'feature1' in drift
    assert 'feature2' in drift

def test_set_baseline():
    """Test setting baseline"""
    tracker = PerformanceTracker()

    baseline_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

    tracker.set_baseline(baseline_data)
    assert tracker.baseline_stats is not None
