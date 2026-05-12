"""MLflow model registry utilities."""

import mlflow
from typing import Optional


class MLflowRegistry:
    """Manage models in MLflow registry."""

    def __init__(self) -> None:
        """Initialize MLflow registry manager."""
        self.client = mlflow.MlflowClient()

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[dict] = None
    ) -> None:
        """
        Register a model.

        Args:
            model_uri: URI of model to register (e.g., runs:/123/model)
            model_name: Name for registered model
            tags: Optional dictionary of tags
        """
        try:
            # Try to create new registered model
            mv = mlflow.register_model(model_uri, model_name)
        except mlflow.exceptions.MlflowException:
            # Model already registered, create new version
            mv = mlflow.register_model(model_uri, model_name)

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name,
                    mv.version,
                    key,
                    value
                )

    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str
    ) -> None:
        """
        Transition model to new stage.

        Args:
            model_name: Name of registered model
            version: Version number
            stage: Target stage (Staging, Production, Archived)

        Raises:
            ValueError: If stage is not valid
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")

        self.client.transition_model_version_stage(
            model_name,
            version,
            stage
        )

    def get_latest_version(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Optional[dict]:
        """
        Get latest model version for stage.

        Args:
            model_name: Name of registered model
            stage: Stage to query (default: Production)

        Returns:
            Model version details or None
        """
        versions = self.client.search_model_versions(
            f"name='{model_name}'"
        )

        for version in versions:
            if version.current_stage == stage:
                return {
                    'version': version.version,
                    'stage': version.current_stage,
                    'model_uri': version.source
                }

        return None
