"""Base model wrapper class."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import Dict, Any, Optional


class BaseModel:
    """Unified interface for classification models."""

    ALGORITHMS = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'xgboost': xgb.XGBClassifier,
    }

    def __init__(
        self,
        algorithm: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize model.

        Args:
            algorithm: Algorithm name
            params: Hyperparameters dictionary

        Raises:
            ValueError: If algorithm not supported
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Algorithm must be one of {list(self.ALGORITHMS.keys())}"
            )

        self.algorithm = algorithm
        self.params = params or {}
        self.model = self.ALGORITHMS[algorithm](**self.params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train model.

        Args:
            X: Training features
            y: Training target
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict on

        Returns:
            Probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Hyperparameters dictionary
        """
        return self.model.get_params()
