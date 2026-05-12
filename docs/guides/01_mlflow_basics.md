# MLflow Basics Guide

## What is MLflow?

MLflow is an open-source platform for managing the machine learning lifecycle. It provides tools for:

1. **Experiment Tracking** - Log and compare experiments
2. **Model Registry** - Manage model versions and lifecycle
3. **Model Serving** - Deploy models in production
4. **Project Execution** - Reproducible ML workflows

## Key Concepts

### Experiments
A container for related runs. Think of it as a project or research area.

### Runs
Individual training jobs within an experiment. Each run logs parameters, metrics, and models.

### MLflow Tracking Server
Logs experiment data locally or to a remote server.

### Model Registry
Centralized repository for managing model versions with staging (Dev → Staging → Production).

## Quick Start with Our Project

### 1. Initialize Tracking
```python
from src.mlflow_integration.tracker import MLflowTracker

tracker = MLflowTracker(experiment_name="dev_burnout")
tracker.start_run(run_name="logistic_regression_v1")
```

### 2. Log Parameters
```python
tracker.log_params({
    'learning_rate': 0.01,
    'max_iter': 1000,
    'solver': 'lbfgs'
})
```

### 3. Log Metrics
```python
tracker.log_metrics({
    'accuracy': 0.95,
    'precision': 0.93,
    'recall': 0.91,
    'f1_score': 0.92,
    'auc': 0.98
})
```

### 4. Log Model
```python
tracker.log_model(model, artifact_path="model", model_format="sklearn")
```

### 5. End Run
```python
tracker.end_run()
```

## Viewing Experiments

Start the MLflow UI:
```bash
mlflow ui --backend-store-uri file:./models/mlruns
```

Then visit http://localhost:5000

You can:
- Compare different experiments
- View metrics over time
- Download model artifacts
- See parameter differences

## Best Practices

1. **Use descriptive run names** - Include model type and version
2. **Log all parameters** - Reproducibility requires full hyperparameter tracking
3. **Log multiple metrics** - Not just accuracy; include precision, recall, F1
4. **Track time** - Log execution time for performance analysis
5. **Use tags** - Add tags like 'production', 'experiment', 'baseline'

## Common Workflow

```python
# 1. Prepare sources
train_data, test_data = split_data(df)

# 2. Start tracking
tracker = MLflowTracker("experiment_name")
tracker.start_run(run_name="attempt_1")

# 3. Log hyperparameters
tracker.log_params(hyperparams)

# 4. Train model
model.fit(train_data)

# 5. Evaluate
metrics = evaluate(model, test_data)

# 6. Log metrics and model
tracker.log_metrics(metrics)
tracker.log_model(model)

# 7. Done
tracker.end_run()
```

## Next Steps

- Read: [Experiment Tracking Guide](02_experiment_tracking.md)
- Read: [Architecture Overview](../ARCHITECTURE.md)
- Try: Run the Developer Burnout notebook
