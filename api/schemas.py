"""Pydantic models for API requests/responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: List[float] = Field(..., min_items=1)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    features: List[List[float]] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: int
    probability: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[int]
    probabilities: List[float]
    count: int
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    stage: str
    algorithm: str
