# REST API Documentation

## Overview

The FastAPI service provides REST endpoints for model prediction and information retrieval.

**Base URL**: `http://localhost:8000`

**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)

## Getting Started

### 1. Start the Server
```bash
python -m api.main
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Test Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Purpose**: Check if the API is running and model is loaded

**Request**: None

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Example**:
```bash
curl -X GET http://localhost:8000/health
```

---

### 2. Model Information

**Endpoint**: `GET /model_info`

**Purpose**: Get information about the current production model

**Request**: None

**Response**:
```json
{
  "model_name": "ds_model",
  "version": "1",
  "stage": "Production",
  "algorithm": "logistic_regression"
}
```

**Example**:
```bash
curl -X GET http://localhost:8000/model_info
```

---

### 3. Single Prediction

**Endpoint**: `POST /predict`

**Purpose**: Make a single prediction for one sample

**Request**:
```json
{
  "features": [25, 40000, 5, 60, 0.5]
}
```

**Parameters**:
- `features` (array of numbers): Input features in the same order as training data

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "timestamp": "2026-05-12T10:30:45.123456"
}
```

**Fields**:
- `prediction`: Predicted class (0 or 1)
- `probability`: Confidence score (0.0-1.0)
- `timestamp`: ISO format timestamp of prediction

**Examples**:

Using curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [25, 40000, 5, 60, 0.5]}'
```

Using Python:
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'features': [25, 40000, 5, 60, 0.5]}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']:.2%}")
```

Using JavaScript:
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({features: [25, 40000, 5, 60, 0.5]})
})
.then(r => r.json())
.then(data => console.log(`Prediction: ${data.prediction}`));
```

---

### 4. Batch Predictions

**Endpoint**: `POST /predict_batch`

**Purpose**: Make predictions for multiple samples

**Request**:
```json
{
  "features": [
    [25, 40000, 5, 60, 0.5],
    [30, 50000, 8, 50, 0.3],
    [45, 80000, 15, 55, 0.7]
  ]
}
```

**Parameters**:
- `features` (array of arrays): Multiple samples

**Response**:
```json
{
  "predictions": [1, 0, 1],
  "probabilities": [0.85, 0.42, 0.91],
  "count": 3,
  "timestamp": "2026-05-12T10:30:45.123456"
}
```

**Fields**:
- `predictions`: Array of predicted classes
- `probabilities`: Array of confidence scores
- `count`: Number of predictions made
- `timestamp`: ISO format timestamp

**Examples**:

Using curl:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [25, 40000, 5, 60, 0.5],
      [30, 50000, 8, 50, 0.3]
    ]
  }'
```

Using Python:
```python
import requests

response = requests.post(
    'http://localhost:8000/predict_batch',
    json={
        'features': [
            [25, 40000, 5, 60, 0.5],
            [30, 50000, 8, 50, 0.3]
        ]
    }
)

result = response.json()
for i, (pred, prob) in enumerate(zip(result['predictions'], result['probabilities'])):
    print(f"Sample {i+1}: {pred} (confidence: {prob:.2%})")
```

---

## Error Handling

### Model Not Loaded

**Status Code**: 500

**Response**:
```json
{
  "detail": "Model not loaded"
}
```

**Cause**: The model failed to load on startup

**Solution**: Check logs and restart the service

### Invalid Request Format

**Status Code**: 422

**Response**:
```json
{
  "detail": [
    {
      "loc": ["body", "features"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Cause**: Request doesn't match expected schema

**Solution**: Verify request format matches documentation

### Bad Input Data

**Status Code**: 400

**Response**:
```json
{
  "detail": "Error: invalid data format"
}
```

**Cause**: Features have wrong dimensions or type

**Solution**: Verify feature count and types match training data

---

## Feature Requirements

The model expects numeric features in this order:
1. age (numeric)
2. salary (numeric)
3. experience_years (numeric)
4. hours_per_week (numeric)
5. department_encoded (numeric, 0-3)

## Rate Limiting

No rate limiting is currently implemented. For production, consider:
- Implement request throttling
- Add authentication
- Monitor API usage

## Example Workflow

```python
# 1. Check if service is running
requests.get('http://localhost:8000/health')

# 2. Get model info
requests.get('http://localhost:8000/model_info')

# 3. Make single prediction
pred = requests.post('http://localhost:8000/predict',
    json={'features': [30, 50000, 8, 50, 1]})

# 4. Make batch predictions
batch = requests.post('http://localhost:8000/predict_batch',
    json={'features': [
        [30, 50000, 8, 50, 1],
        [25, 40000, 5, 60, 2]
    ]})

# 5. Process results
for features, prediction, probability in zip(
    batch.json()['features'],
    batch.json()['predictions'],
    batch.json()['probabilities']
):
    print(f"{features} → {prediction} ({probability:.2%})")
```

## Troubleshooting

### Connection Refused
```
Error: Connection refused at localhost:8000
```
**Solution**: Start the server with `python -m api.main`

### Timeout
```
Error: Request timeout
```
**Solution**: Check server logs and model loading time

### Model not Found
```
Error: Model not loaded
```
**Solution**: Check MLflow model registry and restart service

## See Also

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html)
- [OpenAPI Specification](https://www.openapis.org/)
