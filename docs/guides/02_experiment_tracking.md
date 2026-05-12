# Experiment Tracking Workflow

## Why Track Experiments?

- **Reproducibility** - Recreate any result
- **Comparison** - Which approach is best?
- **Governance** - Who trained what? When?
- **Debugging** - What changed between runs?

## Tracking Strategy

### 1. Organize by Experiment

Group related runs under one experiment:
```python
tracker = MLflowTracker(experiment_name="burnout_classification")
tracker.start_run(run_name="baseline_lr")
# ... train ...
tracker.end_run()

tracker.start_run(run_name="tuned_lr_v1")
# ... train with different params ...
tracker.end_run()
```

### 2. Log Comprehensive Metrics

Don't just log accuracy. Track:
```python
metrics = {
    'accuracy': score,      # Overall correctness
    'precision': precision, # False positive rate
    'recall': recall,       # False negative rate
    'f1': f1_score,        # Harmonic mean
    'auc': auc_score,      # ROC AUC
    'train_time': elapsed   # Performance metric
}
tracker.log_metrics(metrics)
```

### 3. Log Hyperparameters

Every parameter that affects the model:
```python
tracker.log_params({
    'algorithm': 'logistic_regression',
    'max_iter': 1000,
    'solver': 'lbfgs',
    'penalty': 'l2',
    'C': 1.0,
    'train_size': 0.8,
    'cv_folds': 5,
    'random_state': 42
})
```

### 4. Compare Runs

In MLflow UI, select runs and click "Compare" to see:
- Side-by-side parameters
- Metrics differences
- Which hyperparameters affected performance

### 5. Register Best Model

Once you find a good model:
```python
from src.mlflow_integration.registry import MLflowRegistry

registry = MLflowRegistry()
model_uri = f"runs:/{run_id}/model"
registry.register_model(model_uri, "dev_burnout_classifier")
registry.transition_stage("dev_burnout_classifier", 1, "Staging")
```

## Advanced Topics

### Cross-Validation Tracking

Log metrics at each fold:
```python
for fold in range(cv_folds):
    metrics = evaluate(model, fold_data)
    tracker.log_metrics(metrics, step=fold)
```

### Hyperparameter Sweeps

Log each parameter combination:
```python
for lr in [0.001, 0.01, 0.1]:
    for max_iter in [100, 1000]:
        tracker.start_run(run_name=f"lr_{lr}_iter_{max_iter}")
        tracker.log_params({'learning_rate': lr, 'max_iter': max_iter})
        # train and evaluate...
        tracker.end_run()
```

### Comparing Algorithms

Organize by algorithm:
```python
algorithms = ['logistic_regression', 'random_forest', 'xgboost']
for algo in algorithms:
    tracker.start_run(run_name=algo)
    model = train_algorithm(algo)
    tracker.log_metrics(evaluate(model))
    tracker.log_model(model)
    tracker.end_run()
```

## Dashboard Interpretation

When viewing experiments in MLflow:

1. **Scatter Plot** - Each point is a run, colored by metric
2. **Line Plot** - How metrics changed across runs
3. **Parallel Coordinates** - Visualize high-dimensional parameter space
4. **Confusion Matrix** - Only available if logged as artifact

## Workflow Example

```python
# Dataset: Developer Burnout
dataset = "developer_burnout"
tracker = MLflowTracker(experiment_name=f"{dataset}_classification")

# Try multiple approaches
approaches = {
    'logistic_regression': {'max_iter': 1000},
    'random_forest': {'n_estimators': 100, 'max_depth': 10},
    'xgboost': {'n_estimators': 100, 'learning_rate': 0.1}
}

for approach, params in approaches.items():
    tracker.start_run(run_name=approach)
    tracker.log_params(params)
    
    model = train(approach, params)
    metrics = evaluate(model)
    tracker.log_metrics(metrics)
    tracker.log_model(model)
    
    tracker.end_run()

# View results in MLflow UI
# $ mlflow ui --backend-store-uri file:./models/mlruns
```
