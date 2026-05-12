"""Model training orchestration."""

import numpy as np
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Any
from src.models.base import BaseModel


class ModelTrainer:
    """Train and evaluate models."""

    def __init__(self, cv: int = 5) -> None:
        """
        Initialize trainer.

        Args:
            cv: Number of cross-validation folds
        """
        self.cv = cv

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm: str,
        params: Dict[str, Any]
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train single model.

        Args:
            X: Training features
            y: Training target
            algorithm: Algorithm name
            params: Hyperparameters

        Returns:
            Tuple of (fitted model, metrics dict)
        """
        model = BaseModel(algorithm=algorithm, params=params)
        model.fit(X, y)

        # Calculate cross-validation score
        cv_scores = cross_val_score(
            model.model,
            X, y,
            cv=self.cv,
            scoring='accuracy'
        )

        metrics = {
            'accuracy': model.model.score(X, y),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }

        return model, metrics

    def train_multiple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithms: List[str],
        params_dict: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Train multiple models.

        Args:
            X: Training features
            y: Training target
            algorithms: List of algorithm names
            params_dict: Dictionary of params per algorithm

        Returns:
            List of result dictionaries
        """
        results = []

        for algo in algorithms:
            params = params_dict.get(algo, {})
            model, metrics = self.train(X, y, algo, params)

            results.append({
                'algorithm': algo,
                'model': model,
                'params': params,
                'metrics': metrics
            })

        return results
