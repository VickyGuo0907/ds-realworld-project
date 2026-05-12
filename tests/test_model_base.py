"""Tests for base model wrapper."""

import pytest
import numpy as np
from src.models.base import BaseModel


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_fit(sample_data):
    """Test model fitting."""
    X, y = sample_data
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})
    model.fit(X, y)

    assert model.is_fitted


def test_predict(sample_data):
    """Test prediction."""
    X, y = sample_data
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})
    model.fit(X, y)
    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    assert all(p in [0, 1] for p in predictions)


def test_predict_proba(sample_data):
    """Test probability predictions."""
    X, y = sample_data
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})
    model.fit(X, y)
    probs = model.predict_proba(X[:10])

    assert probs.shape == (10, 2)
    assert np.all((probs >= 0) & (probs <= 1))


def test_invalid_algorithm():
    """Test invalid algorithm raises ValueError."""
    with pytest.raises(ValueError):
        BaseModel(algorithm="invalid_algorithm")


def test_predict_without_fit(sample_data):
    """Test prediction without fitting raises ValueError."""
    X, _ = sample_data
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})

    with pytest.raises(ValueError):
        model.predict(X[:10])


def test_predict_proba_without_fit(sample_data):
    """Test probability prediction without fitting raises ValueError."""
    X, _ = sample_data
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})

    with pytest.raises(ValueError):
        model.predict_proba(X[:10])


def test_get_params(sample_data):
    """Test getting model parameters."""
    model = BaseModel(algorithm="logistic_regression", params={"max_iter": 1000})
    params = model.get_params()

    assert isinstance(params, dict)
    assert "max_iter" in params
    assert params["max_iter"] == 1000


def test_supported_algorithms():
    """Test all supported algorithms can be initialized."""
    algorithms = ["logistic_regression", "random_forest", "xgboost"]

    for algo in algorithms:
        model = BaseModel(algorithm=algo)
        assert model.algorithm == algo
        assert model.is_fitted is False
