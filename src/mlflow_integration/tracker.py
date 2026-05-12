"""MLflow experiment tracking utilities."""

import mlflow
from typing import Dict, Any, Optional


class MLflowTracker:
    """Track experiments with MLflow."""

    def __init__(
        self, experiment_name: str, tracking_uri: str = "file:./models/mlruns"
    ) -> None:
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        except AttributeError:
            experiment_id = mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id

    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for run
        """
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(
        self, model: Any, artifact_path: str = "model", model_format: str = "sklearn"
    ) -> None:
        """
        Log model artifact.

        Args:
            model: Trained model
            artifact_path: Path to save model
            model_format: Format (sklearn, tensorflow, etc)
        """
        if model_format == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        elif model_format == "tensorflow":
            mlflow.tensorflow.log_model(model, artifact_path=artifact_path)

    def log_artifact(self, local_path: str) -> None:
        """
        Log artifact file.

        Args:
            local_path: Path to artifact file
        """
        mlflow.log_artifact(local_path)

    def end_run(self) -> None:
        """End current MLflow run."""
        mlflow.end_run()

    def get_run_id(self) -> str:
        """
        Get current run ID.

        Returns:
            Run ID string
        """
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError("No active run")
        return active.info.run_id
