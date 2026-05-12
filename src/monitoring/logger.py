"""Logging for predictions."""

import json
import logging
from datetime import datetime
from typing import Optional, List


class PredictionLogger:
    """Log predictions for monitoring."""

    def __init__(self, log_file: str = 'predictions.log') -> None:
        """
        Initialize logger.

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger('predictions')

        # Clear existing handlers
        logger.handlers = []

        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def log_prediction(
        self,
        features: List[float],
        prediction: int,
        probability: float,
        true_label: Optional[int] = None
    ) -> None:
        """
        Log a prediction.

        Args:
            features: Input features
            prediction: Predicted label
            probability: Prediction confidence
            true_label: True label if available
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'probability': probability,
            'true_label': true_label
        }

        self.logger.info(json.dumps(log_entry))
