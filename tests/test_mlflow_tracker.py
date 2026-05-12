import pytest
import mlflow
from src.mlflow_integration.tracker import MLflowTracker


@pytest.fixture
def tracker():
    """Create MLflow tracker instance"""
    mlflow.set_tracking_uri("file:./models/mlruns")
    tracker = MLflowTracker(experiment_name="test_experiment")
    yield tracker
    mlflow.end_run()


def test_start_run(tracker):
    """Test starting MLflow run"""
    tracker.start_run(run_name="test_run")

    assert mlflow.active_run() is not None
    assert mlflow.active_run().info.run_name == "test_run"


def test_log_params(tracker):
    """Test logging parameters"""
    tracker.start_run()
    tracker.log_params({"learning_rate": 0.01, "max_depth": 10})

    assert mlflow.active_run() is not None


def test_log_metrics(tracker):
    """Test logging metrics"""
    tracker.start_run()
    tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.93})

    assert mlflow.active_run() is not None


def test_end_run(tracker):
    """Test ending run"""
    tracker.start_run()
    tracker.end_run()

    assert mlflow.active_run() is None
