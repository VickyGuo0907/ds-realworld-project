# System Architecture

## Project Overview

```
Data (Kaggle) 
    ↓
Data Pipeline (src/data/) 
    ├→ Load CSV
    ├→ Preprocess & Clean
    └→ Split Train/Test
    ↓
Feature Engineering (src/features/)
    └→ Create New Features
    ↓
Model Training (src/models/)
    ├→ Logistic Regression
    ├→ Random Forest
    └→ XGBoost
    ↓
MLflow Integration (src/mlflow_integration/)
    ├→ Track Experiments
    └→ Manage Versions
    ↓
Evaluation (src/evaluation/)
    └→ Metrics & Comparison
    ↓
FastAPI Service (src/api/)
    └→ REST Endpoints
    ↓
Monitoring (src/monitoring/)
    └→ Predictions & Drift
```

## Directory Structure

```
ds_realworld_project/
├── src/                          # Production library
│   ├── data/                    # Data pipeline
│   │   ├── loader.py            # CSV loading
│   │   ├── preprocessor.py      # Data cleaning
│   │   └── splitter.py          # Train/test splits
│   ├── features/                # Feature engineering
│   │   └── engineer.py          # Feature creation
│   ├── models/                  # Model training
│   │   ├── base.py              # Unified interface
│   │   └── trainer.py           # Training orchestration
│   ├── evaluation/              # Model evaluation
│   │   └── metrics.py           # Metric calculations
│   ├── mlflow_integration/      # MLflow utilities
│   │   ├── tracker.py           # Experiment tracking
│   │   └── registry.py          # Model versioning
│   ├── monitoring/              # Monitoring
│   │   ├── logger.py            # Prediction logging
│   │   └── performance.py       # Drift detection
│   ├── api/                     # FastAPI service
│   │   ├── main.py              # REST endpoints
│   │   └── schemas.py           # Request/response models
│   ├── images/                  # Project images & diagrams
│   └── notebooks/               # Educational notebooks
│       ├── 1_developer_burnout_pipeline.ipynb
│       └── 2_teen_mental_health_complete_pipeline.ipynb
├── tests/                       # Unit & integration tests
├── docs/                        # Documentation
│   ├── guides/                 # Learning guides
│   └── ARCHITECTURE.md         # This file
├── sources/
│   ├── raw/                    # Kaggle datasets
│   └── processed/              # Cleaned data
├── models/                      # Model artifacts
│   └── mlruns/                 # MLflow artifacts
├── config.yaml                 # Configuration
└── pyproject.toml              # Dependencies
```

## Module Responsibilities

### src/data/ - Data Pipeline
- **loader.py**: Load datasets from CSV with validation
- **preprocessor.py**: Handle missing values, encode categories, normalize
- **splitter.py**: Split data with stratification

**Inputs**: Raw CSV files from Kaggle  
**Outputs**: Train/test DataFrames ready for modeling

### src/features/ - Feature Engineering
- **engineer.py**: Create polynomial, interaction, and ratio features

**Inputs**: Preprocessed DataFrames  
**Outputs**: Enhanced DataFrames with new features

### src/models/ - Model Training
- **base.py**: Unified interface for all algorithms
- **trainer.py**: Training with cross-validation

**Inputs**: Feature matrices and targets  
**Outputs**: Trained models and metrics

**Algorithms Supported**:
- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)

### src/evaluation/ - Model Evaluation
- **metrics.py**: Calculate classification metrics

**Inputs**: Predictions and true labels  
**Outputs**: Accuracy, precision, recall, F1, AUC

### src/mlflow_integration/ - Experiment Management
- **tracker.py**: Log experiments to MLflow
- **registry.py**: Manage model versions and staging

**Workflow**:
1. Start run
2. Log parameters and metrics
3. Log model artifact
4. Register model to registry
5. Transition to production

### src/api/ - REST Service
- **main.py**: FastAPI endpoints
- **schemas.py**: Pydantic request/response models

**Endpoints**:
- `GET /health` - Health check
- `GET /model_info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

### src/monitoring/ - Production Monitoring
- **logger.py**: Log all predictions for audit trail
- **performance.py**: Detect data and prediction drift

## Data Flow

```
1. Load: CSV → DataFrame
2. Preprocess: Clean, encode, normalize
3. Features: Polynomial, interactions, ratios
4. Split: Train/validation/test sets
5. Train: Multiple algorithms with CV
6. Evaluate: Metrics and comparison
7. Register: Best model to MLflow registry
8. Serve: REST API loads model, makes predictions
9. Monitor: Log predictions, track drift
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data | pandas, numpy, scikit-learn |
| ML | scikit-learn, XGBoost, TensorFlow |
| Tracking | MLflow |
| Serving | FastAPI, Pydantic |
| Testing | pytest |
| Docs | Jupyter, Markdown |

## Key Design Decisions

### 1. Modular Architecture
Each module has one responsibility. Easy to test, reuse, and modify.

### 2. TDD Approach
Tests written first. Ensures correctness and documents behavior.

### 3. Unified Model Interface
BaseModel class supports multiple algorithms with same API.

### 4. MLflow Integration
Automatic experiment tracking. Reproducible and governable.

### 5. Educational First
Clear docstrings, type hints, and comprehensive tests.

## Deployment Path

```
Development
    ↓ (Local MLflow + Jupyter)
Staging
    ↓ (FastAPI + MLflow registry)
Production
    ↓ (Containerized, monitored)
```

## Performance Considerations

- **Data Loading**: Lazy loaded, handled in chunks
- **Model Training**: Cross-validation for robustness
- **Prediction**: ~10ms per prediction with FastAPI
- **Monitoring**: Asynchronous logging to avoid latency

## Scalability

Current design handles:
- **Data Size**: Up to 100k rows (in-memory)
- **Feature Dimensionality**: Typical ML datasets
- **Concurrent Requests**: FastAPI handles ~100 req/s on standard hardware

For larger scale, consider:
- Distributed training (Spark)
- Model serving (KServe, BentoML)
- Stream processing (Kafka)
