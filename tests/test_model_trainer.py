"""Tests for model trainer."""

import pytest
import numpy as np
from src.models.trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample sources."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_train_single_model(sample_data):
    """Test training single model."""
    X, y = sample_data
    trainer = ModelTrainer()

    model, metrics = trainer.train(
        X, y, algorithm="logistic_regression", params={"max_iter": 1000}
    )

    assert model.is_fitted
    assert "accuracy" in metrics


def test_train_multiple_models(sample_data):
    """Test training multiple models."""
    X, y = sample_data
    trainer = ModelTrainer()

    results = trainer.train_multiple(
        X,
        y,
        algorithms=["logistic_regression", "random_forest"],
        params_dict={
            "logistic_regression": {"max_iter": 1000},
            "random_forest": {"n_estimators": 10},
        },
    )

    assert len(results) == 2
    assert all("metrics" in r for r in results)


def test_cv_folds_parameter():
    """Test custom cv folds parameter."""
    trainer = ModelTrainer(cv=10)
    assert trainer.cv == 10


def test_metrics_content(sample_data):
    """Test metrics dictionary contains expected keys."""
    X, y = sample_data
    trainer = ModelTrainer()

    model, metrics = trainer.train(
        X, y, algorithm="logistic_regression", params={"max_iter": 1000}
    )

    assert "accuracy" in metrics
    assert "cv_mean" in metrics
    assert "cv_std" in metrics
    assert isinstance(metrics["accuracy"], (int, float))
    assert isinstance(metrics["cv_mean"], (int, float))
    assert isinstance(metrics["cv_std"], (int, float))


def test_train_multiple_returns_correct_structure(sample_data):
    """Test train_multiple returns correct structure."""
    X, y = sample_data
    trainer = ModelTrainer()

    results = trainer.train_multiple(
        X,
        y,
        algorithms=["logistic_regression"],
        params_dict={"logistic_regression": {"max_iter": 1000}},
    )

    assert len(results) == 1
    result = results[0]
    assert "algorithm" in result
    assert "model" in result
    assert "params" in result
    assert "metrics" in result
    assert result["algorithm"] == "logistic_regression"


def test_train_with_empty_params(sample_data):
    """Test train with empty parameters."""
    X, y = sample_data
    trainer = ModelTrainer()

    model, metrics = trainer.train(X, y, algorithm="logistic_regression", params={})

    assert model.is_fitted
    assert "accuracy" in metrics


def test_train_multiple_with_different_algorithms(sample_data):
    """Test train_multiple with various algorithms."""
    X, y = sample_data
    trainer = ModelTrainer()

    results = trainer.train_multiple(
        X,
        y,
        algorithms=["logistic_regression", "random_forest", "xgboost"],
        params_dict={
            "logistic_regression": {"max_iter": 1000},
            "random_forest": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10},
        },
    )

    assert len(results) == 3
    algos = [r["algorithm"] for r in results]
    assert algos == ["logistic_regression", "random_forest", "xgboost"]
