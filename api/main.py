"""FastAPI application for model serving."""

from fastapi import FastAPI, HTTPException
from datetime import datetime
import numpy as np
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
)

app = FastAPI(title="DS Pipeline API", version="1.0.0")

# Global model cache
_model_metadata = None


def load_model():
    """Load model metadata on startup."""
    global _model_metadata

    _model_metadata = {
        "model_name": "ds_model",
        "version": "1",
        "stage": "Production",
        "algorithm": "logistic_regression",
    }


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _model_metadata is not None}


@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info() -> dict:
    """Get current model information."""
    if _model_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return _model_metadata


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> dict:
    """Make single prediction."""
    if _model_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        np.array(request.features).reshape(1, -1)
        # Dummy prediction for now (would load real model from MLflow)
        prediction = 1
        probability = 0.85

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> dict:
    """Make batch predictions."""
    if _model_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        np.array(request.features)
        batch_size = len(request.features)

        # Dummy predictions (would use real model from MLflow)
        predictions = [0] * batch_size
        probabilities = [0.5] * batch_size

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": batch_size,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
