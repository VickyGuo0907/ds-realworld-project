---
name: MLflow Educational Pipeline Design
description: Comprehensive DS learning project with MLflow integration, multiple datasets, and multi-stage deployment
type: design
date: 2026-05-11
---

# MLflow Educational Pipeline: Design Specification

## 1. Project Overview

**Goal**: Build a comprehensive, educational data science project that teaches others the complete ML lifecycle - from raw data to production-ready model serving - using real Kaggle datasets, MLflow for experiment tracking and model management, and FastAPI for serving.

**Target Audience**: Data scientists and engineers learning:
- End-to-end ML pipeline development
- Experiment tracking and model versioning with MLflow
- Multiple algorithm comparisons and selection
- Production-ready code practices
- Multi-stage deployment (dev → staging → production)

**Datasets**:
1. Developer Burnout (classification)
2. Tennessee Mental Health (classification)

**Key Constraint**: Educational focus - code and notebooks should explain "why" as much as "what".

---

## 2. Architecture Overview

### 2.1 High-Level Structure

```
ds_realworld_project/
├── src/                          # Reusable ML library (modular)
│   ├── data/                    # Data pipeline
│   │   ├── loader.py            # Load datasets from CSV
│   │   ├── preprocessor.py      # Data cleaning, validation
│   │   └── splitter.py          # Train/test/val splits
│   ├── features/                # Feature engineering
│   │   ├── engineer.py          # Feature creation, transformation
│   │   └── selectors.py         # Feature selection utilities
│   ├── models/                  # Model training
│   │   ├── base.py              # Base model wrapper class
│   │   ├── algorithms.py        # Algorithm implementations (LR, RF, XGB, NN)
│   │   └── trainer.py           # Training orchestration
│   ├── evaluation/              # Model evaluation
│   │   ├── metrics.py           # Metric calculations
│   │   └── comparator.py        # Algorithm comparison utilities
│   ├── mlflow_integration/      # MLflow utilities
│   │   ├── tracker.py           # Experiment tracking (log params, metrics, models)
│   │   ├── registry.py          # Model registry operations
│   │   └── config.py            # MLflow configuration
│   ├── serving/                 # Model serving
│   │   ├── predictor.py         # Load and predict with MLflow models
│   │   └── api.py               # FastAPI application
│   └── monitoring/              # Production monitoring
│       ├── logger.py            # Log predictions for monitoring
│       └── performance.py       # Track model drift/performance
├── notebooks/                   # Educational notebooks
│   ├── 1_developer_burnout_pipeline.ipynb
│   │   └── Full pipeline walkthrough: EDA → FE → Models → MLflow → Serving
│   └── 2_tennessee_mental_health_advanced.ipynb
│       └── Advanced concepts: Comparisons, tuning, monitoring
├── api/                         # FastAPI application
│   ├── main.py                 # API endpoints
│   ├── schemas.py              # Pydantic models
│   └── config.py               # API configuration
├── tests/                       # Unit and integration tests
├── docs/                        # Documentation
│   ├── guides/                 # Blog post style guides
│   │   ├── 01_mlflow_basics.md
│   │   ├── 02_experiment_tracking.md
│   │   └── 03_model_serving.md
│   ├── api_docs.md             # API documentation
│   └── learning_path.md        # Curriculum guide
├── models/                      # MLflow artifact storage (local)
│   └── mlruns/                 # MLflow experiment runs
├── data/
│   ├── raw/                     # Original Kaggle datasets
│   └── processed/               # Cleaned and processed
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project metadata
└── main.py                     # Entry point for training pipelines
```

### 2.2 Component Interactions

```
User/Notebook
    ↓
Data Loading (src/data/loader.py)
    ↓
Preprocessing (src/data/preprocessor.py)
    ↓
Feature Engineering (src/features/engineer.py)
    ↓
Model Training (src/models/trainer.py)
    ├→ MLflow Tracking (src/mlflow_integration/tracker.py)
    │   └→ Log params, metrics, models to MLflow
    ↓
Model Evaluation (src/evaluation/metrics.py)
    ├→ MLflow Registry (src/mlflow_integration/registry.py)
    │   └→ Register best model
    ↓
API Serving (api/main.py)
    ├→ Load from MLflow Registry
    └→ FastAPI endpoints for predictions
    ├→ Monitoring (src/monitoring/logger.py)
```

---

## 3. Detailed Components

### 3.1 Data Pipeline (`src/data/`)

**Purpose**: Reusable data loading and preprocessing for both datasets

**Components**:
- `loader.py`: Load CSV files with validation
- `preprocessor.py`: Handle missing values, encoding, scaling
- `splitter.py`: Train/test/validation splits with stratification

**MLflow Integration**: Log data version/checksums as run metadata

**Teaching Points**:
- Data validation best practices
- Handling imbalanced datasets
- Cross-validation strategies

### 3.2 Feature Engineering (`src/features/`)

**Purpose**: Create and select features for models

**Components**:
- `engineer.py`: Feature creation, transformations, domain knowledge
- `selectors.py`: Feature importance analysis, selection methods

**Teaching Points**:
- Why feature engineering matters
- Different feature selection approaches
- Domain knowledge vs statistical methods

### 3.3 Model Training (`src/models/`)

**Purpose**: Train multiple algorithms and compare

**Algorithms**:
1. Logistic Regression (baseline)
2. Random Forest (ensemble)
3. XGBoost (gradient boosting)
4. Neural Network (deep learning)

**Components**:
- `base.py`: Unified interface for all models
- `algorithms.py`: Algorithm implementations
- `trainer.py`: Training loop, hyperparameter optimization

**MLflow Integration**:
- Log hyperparameters before training
- Log metrics after training
- Log model artifacts
- Track training time and resources

**Teaching Points**:
- When to use each algorithm
- Hyperparameter tuning strategies (grid search, random search, Optuna)
- Cross-validation and evaluation

### 3.4 Evaluation (`src/evaluation/`)

**Purpose**: Evaluate and compare models

**Metrics by Problem Type** (Classification):
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix, Classification Report

**Components**:
- `metrics.py`: Metric calculations
- `comparator.py`: Side-by-side algorithm comparison

**MLflow Integration**:
- Log all metrics to MLflow
- Create comparison visualizations

**Teaching Points**:
- Which metrics matter for different problems
- ROC vs PR curves
- Bias-variance tradeoff

### 3.5 MLflow Integration (`src/mlflow_integration/`)

**Purpose**: Centralized MLflow operations

**Components**:
- `tracker.py`: Log experiments, metrics, parameters, models
- `registry.py`: Register models, transition stages (dev → staging → production)
- `config.py`: MLflow configuration (server, tracking URI)

**Workflow**:
1. `tracker.py` logs during training
2. `registry.py` registers best model after evaluation
3. Staging transitions for model governance

**Teaching Points**:
- Experiment tracking workflow
- Model versioning and governance
- MLflow UI for monitoring experiments

### 3.6 FastAPI Service (`api/`)

**Purpose**: REST API for model serving

**Endpoints**:
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /model_info` - Get current model info (version, stage)
- `GET /health` - Health check

**Components**:
- `main.py`: FastAPI application
- `schemas.py`: Pydantic request/response models
- `config.py`: API configuration

**MLflow Integration**:
- Load model from MLflow Model Registry
- Load the "Production" stage model

**Teaching Points**:
- REST API design
- Request/response validation
- Model serving best practices

### 3.7 Monitoring (`src/monitoring/`)

**Purpose**: Track model performance in "production"

**Components**:
- `logger.py`: Log predictions and ground truth
- `performance.py`: Calculate performance metrics, detect drift

**Metrics Tracked**:
- Prediction latency
- Input feature distributions (data drift)
- Model prediction distributions (prediction drift)
- Actual performance vs baseline

**Teaching Points**:
- Why monitoring matters
- Model drift detection
- Retraining triggers

---

## 4. Multi-Stage Deployment

### 4.1 Development Stage

**Environment**: Local machine

**Workflow**:
1. Run notebook to explore data
2. Train models locally
3. Log experiments to local MLflow server
4. Compare models in MLflow UI
5. Register best model to "Staging" stage

**Tools**: Jupyter notebooks, MLflow UI (local), scikit-learn/XGBoost

### 4.2 Staging Stage

**Environment**: Local + FastAPI

**Workflow**:
1. Load model from MLflow "Staging" stage
2. Run FastAPI service locally
3. Test API endpoints
4. Validate predictions on holdout test set
5. Monitor prediction latency and correctness

**Tools**: FastAPI, MLflow Model Registry, Pydantic

### 4.3 Production Stage

**Environment**: Ready for deployment (instructions for cloud deployment)

**Workflow**:
1. Transition model to "Production" in MLflow Registry
2. API service loads "Production" model
3. Log all predictions for monitoring
4. Track data drift and performance
5. Trigger retraining when needed

**Deployment Options Documented**:
- Local (for learning)
- Docker (containerized)
- Cloud (AWS/GCP/Azure with MLflow)

---

## 5. Educational Structure

### 5.1 Two Separate Notebooks

**Notebook 1: Developer Burnout (Core Concepts)**
- Full pipeline walkthrough
- Explains every step
- Single algorithm (Logistic Regression) + Random Forest
- Introduces MLflow
- Introduces FastAPI serving

**Notebook 2: Tennessee Mental Health (Advanced Topics)**
- Applies the library
- Compares all 4 algorithms
- Hyperparameter tuning
- Feature importance analysis
- Model monitoring

### 5.2 Documentation Tiers

**Tier 1: Jupyter Notebooks** (Step-by-step learning)
- Inline markdown explanations
- Visualizations
- Code examples

**Tier 2: Module Docstrings** (Code documentation)
- Clear docstrings explaining purpose
- Type annotations
- Example usage

**Tier 3: Blog Guides** (`docs/guides/`)
- Conceptual explanations
- Why we make certain decisions
- Best practices
- Links to notebooks and code

**Tier 4: API Documentation** (`docs/api_docs.md`)
- Endpoint descriptions
- Request/response examples
- curl examples

### 5.3 Learning Path

**For Learner A** (Notebook-first):
1. Read Developer Burnout notebook
2. Run notebook locally
3. Check MLflow UI to see experiments
4. Read guide docs for conceptual understanding
5. Explore Python modules for implementation details

**For Learner B** (Code-first):
1. Read architecture overview
2. Explore modular Python code
3. Run notebooks to see code in action
4. Run FastAPI service to test predictions
5. Read guides for deeper understanding

---

## 6. Key Features

### 6.1 Experiment Tracking with MLflow

**What's Logged**:
- Hyperparameters (learning rate, tree depth, etc.)
- Metrics (accuracy, F1, AUC for each split)
- Model artifacts (serialized model files)
- Evaluation plots (confusion matrix, ROC curve)
- Execution time and resource usage

**MLflow UI Benefits**:
- Compare experiments side-by-side
- Track metric evolution over runs
- Download model artifacts
- Visual comparison of algorithm performance

### 6.2 Model Versioning

**Workflow**:
1. Train model → logged to MLflow
2. Evaluate → MLflow records metrics
3. Register → `mlflow.register_model()` creates version 1
4. Stage transition → move to "Staging" for testing
5. Promote → move to "Production" when ready

**Teaching**: How to manage model versions and prevent "model soup"

### 6.3 FastAPI Serving

**Features**:
- Type-safe requests/responses (Pydantic)
- Automatic API documentation (Swagger UI)
- Batch prediction support
- Model info endpoint

**Teaching**: REST API design, request validation, error handling

### 6.4 Monitoring Setup

**Tracks**:
- Prediction latency
- Input feature statistics (detect data drift)
- Model prediction distribution (detect prediction drift)
- Actual performance when labels available

**Teaching**: Why production monitoring matters

---

## 7. Technology Stack

### Core Libraries
- **MLflow**: Experiment tracking, model registry, model serving
- **FastAPI**: REST API framework
- **Pydantic**: Data validation
- **scikit-learn**: Logistic Regression, Random Forest
- **XGBoost**: Gradient boosting
- **TensorFlow/PyTorch**: Neural networks (optional, simplified)

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Preprocessing, feature engineering

### Visualization & Analysis
- **matplotlib, seaborn**: Plotting
- **plotly**: Interactive visualizations
- **shap**: Feature importance (SHAP values)

### Testing & Quality
- **pytest**: Unit testing
- **pytest-cov**: Code coverage
- **black**: Code formatting
- **ruff**: Linting

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Data loading and preprocessing
- Feature engineering functions
- Metric calculations
- API endpoint validation

### 8.2 Integration Tests
- End-to-end pipeline (data → model → evaluation)
- MLflow registration and retrieval
- FastAPI endpoints with real models

### 8.3 Coverage Target
- Minimum 80% code coverage in `src/`
- All public functions tested
- Edge cases and error conditions covered

---

## 9. Development Phases

### Phase 1: Foundation (Week 1)
- Setup project structure
- Implement data pipeline for both datasets
- Create feature engineering module
- Write tests for data and feature modules

### Phase 2: Modeling (Week 2)
- Implement all 4 algorithms
- MLflow integration for training
- Model evaluation and comparison
- Notebook 1 (Developer Burnout) - core concepts

### Phase 3: Serving & Deployment (Week 3)
- FastAPI application
- MLflow Model Registry integration
- Test API endpoints
- Documentation and guides

### Phase 4: Monitoring & Advanced (Week 4)
- Monitoring module
- Notebook 2 (Tennessee Mental Health) - advanced topics
- Blog post guides
- Learning path documentation

---

## 10. Documentation Deliverables

### 10.1 Code Documentation
- Clear docstrings (all functions)
- Type annotations
- Usage examples in docstrings

### 10.2 Notebooks
- Markdown explanations between code cells
- Visualizations with interpretations
- "Why" explanations, not just "how"

### 10.3 Blog Guides
- MLflow basics
- Experiment tracking workflow
- Model serving with FastAPI
- Monitoring and drift detection
- When to use each algorithm

### 10.4 README & Learning Path
- Project overview
- Architecture diagram
- Multiple entry points (notebook-first vs code-first)
- Links to guides and notebooks

---

## 11. Success Criteria

✅ **Learner can**:
1. Understand the full ML pipeline end-to-end
2. Run notebooks locally and see results
3. Compare multiple algorithms systematically
4. Track experiments in MLflow
5. Serve models via REST API
6. Monitor model performance
7. Implement similar pipeline on new dataset

✅ **Code quality**:
- 80%+ test coverage
- All functions have docstrings
- Max 40 lines per function
- Clear module boundaries

✅ **Documentation**:
- Multiple learning paths (notebook-first, code-first)
- Blog guides explaining concepts
- API documentation
- Architecture diagrams

---

## 12. Open Questions / Refinements Needed

None - design is comprehensive and ready for implementation.

---

## 13. Implementation Dependencies

**Before Implementation**:
1. Ensure both datasets are loaded and available in `data/raw/`
2. Decide final algorithm choices (e.g., which NN framework: TensorFlow vs PyTorch?)
3. Decide MLflow deployment: local file storage vs local server
4. Finalize hyperparameter tuning strategy (grid vs random vs Optuna)

**After Design Approval**:
- Use planning skill to create detailed implementation plan with tasks
