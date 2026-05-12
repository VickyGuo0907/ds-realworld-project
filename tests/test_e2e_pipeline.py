"""End-to-end pipeline integration test.

This test validates the complete ML pipeline from data loading through
evaluation and MLflow tracking.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import EvaluationMetrics
from src.mlflow_integration.tracker import MLflowTracker
import mlflow


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing.

    Creates synthetic data with realistic features for testing the
    complete pipeline without external dependencies.

    Yields:
        Path to temporary CSV file with test data
    """
    np.random.seed(42)
    n_samples = 200

    data = {
        'age': np.random.randint(20, 65, n_samples),
        'income': np.random.randint(30000, 150000, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'hours_per_week': np.random.randint(20, 70, n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Finance'], n_samples),
        'burnout': np.random.randint(0, 2, n_samples)
    }

    df = pd.DataFrame(data)

    # Save to temporary CSV
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_data.csv"
        df.to_csv(csv_path, index=False)
        yield str(csv_path)


@pytest.fixture
def mlflow_experiment():
    """Set up MLflow experiment for testing.

    Yields:
        Experiment name string
    """
    import uuid
    experiment_name = f"e2e_test_experiment_{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri("file:./models/mlruns")

    yield experiment_name


class TestEndToEndPipeline:
    """Test complete ML pipeline end-to-end."""

    def test_complete_ml_pipeline(self, sample_dataset, mlflow_experiment):
        """Test complete ML pipeline from data to evaluation.

        This test validates:
        1. Data loading
        2. Preprocessing (handling missing values, encoding, normalization)
        3. Feature engineering (polynomial and interaction features)
        4. Data splitting (train/test with stratification)
        5. Model training (multiple algorithms)
        6. Model evaluation (metrics calculation)
        7. MLflow experiment tracking

        Args:
            sample_dataset: Path to sample dataset fixture
            mlflow_experiment: MLflow experiment name fixture
        """
        # STEP 1: Load Data
        loader = DataLoader(sample_dataset)
        df = loader.load()

        assert isinstance(df, pd.DataFrame), "Loaded data should be DataFrame"
        assert len(df) == 200, "Sample dataset should have 200 rows"
        assert 'burnout' in df.columns, "Target column 'burnout' should exist"
        assert 'age' in df.columns, "Feature 'age' should exist"

        # STEP 2: Preprocess Data
        numeric_cols = ['age', 'income', 'experience', 'hours_per_week']
        categorical_cols = ['department']

        preprocessor = DataPreprocessor(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )
        df_processed = preprocessor.fit_transform()

        assert df_processed.shape[0] == 200, "Processed data should have 200 rows"
        assert df_processed.isnull().sum().sum() == 0, "No null values after preprocessing"
        assert df_processed[numeric_cols].notna().all().all(), "Numeric cols should not be null"

        # STEP 3: Feature Engineering
        engineer = FeatureEngineer(df_processed)
        df_features = engineer.create_polynomial_features(['age'], degree=2)
        engineer = FeatureEngineer(df_features)
        df_features = engineer.create_interaction_features(['age', 'income'])

        assert 'age_2' in df_features.columns, "Polynomial feature 'age_2' should exist"
        assert 'age_x_income' in df_features.columns, "Interaction feature should exist"
        initial_feature_count = df_features.shape[1]

        # STEP 4: Split Data
        splitter = DataSplitter(df_features, target_col='burnout', test_size=0.2)
        train, test = splitter.train_test_split(stratify=True)

        assert len(train) + len(test) == 200, "Split data should sum to original size"
        assert len(test) == 40, "Test set should be 20% of 200 rows"
        assert len(train) == 160, "Train set should be 80% of 200 rows"

        X_train = train.drop('burnout', axis=1).values
        y_train = train['burnout'].values
        X_test = test.drop('burnout', axis=1).values
        y_test = test['burnout'].values

        assert X_train.shape[0] == 160, "Training set should have 160 samples"
        assert X_test.shape[0] == 40, "Test set should have 40 samples"

        # STEP 5: Train Models
        trainer = ModelTrainer(cv=3)
        results = trainer.train_multiple(
            X_train, y_train,
            algorithms=['logistic_regression', 'random_forest'],
            params_dict={
                'logistic_regression': {'max_iter': 1000},
                'random_forest': {'n_estimators': 10, 'max_depth': 5}
            }
        )

        assert len(results) == 2, "Should train 2 models"
        assert all(r['model'].is_fitted for r in results), "All models should be fitted"
        assert all('metrics' in r for r in results), "All results should have metrics"
        assert all('accuracy' in r['metrics'] for r in results), "Accuracy should be in metrics"

        # STEP 6: Evaluate Models
        best_result = max(results, key=lambda r: r['metrics']['accuracy'])
        predictions = best_result['model'].predict(X_test)

        evaluator = EvaluationMetrics(y_test, predictions)
        metrics = evaluator.calculate_metrics()

        assert 'accuracy' in metrics, "Accuracy should be in metrics"
        assert 'precision' in metrics, "Precision should be in metrics"
        assert 'recall' in metrics, "Recall should be in metrics"
        assert 'f1' in metrics, "F1 score should be in metrics"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"

        cm = evaluator.get_confusion_matrix()
        assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"

        # STEP 7: MLflow Tracking
        tracker = MLflowTracker(experiment_name=mlflow_experiment)
        tracker.start_run(run_name="full_pipeline_e2e")

        tracker.log_params({
            'algorithm': best_result['algorithm'],
            'cv_folds': 3,
            'test_size': 0.2,
            'feature_count': initial_feature_count,
            'train_size': len(train)
        })

        tracker.log_metrics(metrics)

        # Verify we can get run ID
        run_id = tracker.get_run_id()
        assert run_id is not None, "Run ID should not be None"
        assert isinstance(run_id, str), "Run ID should be string"

        tracker.end_run()

        # STEP 8: Verify MLflow
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(mlflow_experiment)
        assert exp is not None, "Experiment should exist in MLflow"

        print("\n✅ End-to-End Pipeline Test PASSED")
        print(f"  Algorithms tested: {len(results)}")
        print(f"  Best model: {best_result['algorithm']}")
        print(f"  Best accuracy: {best_result['metrics']['accuracy']:.4f}")
        print(f"  Test set size: {len(y_test)}")
        print(f"  Features created: {initial_feature_count}")

    def test_data_loading_and_validation(self, sample_dataset):
        """Test data loading with validation.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df, info = loader.load_with_info()

        assert df.shape[0] == 200, "Should load 200 rows"
        assert info['shape'] == (200, 6), "Shape should be (200, 6)"
        assert len(info['columns']) == 6, "Should have 6 columns"
        assert all(v == 0 for v in info['null_counts'].values()), "No null values expected"

        print("\n✅ Data Loading and Validation Test PASSED")

    def test_preprocessing_chain(self, sample_dataset):
        """Test preprocessing pipeline.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        numeric_cols = ['age', 'income', 'experience', 'hours_per_week']
        categorical_cols = ['department']

        preprocessor = DataPreprocessor(
            df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )

        # Test individual preprocessing steps
        df_missing = preprocessor.handle_missing_values(strategy='mean')
        assert df_missing.shape == df.shape, "Shape should not change"

        preprocessor.df = df
        df_encoded = preprocessor.encode_categorical()
        assert df_encoded['department'].dtype in [np.int64, np.int32], "Categorical col should be numeric"

        preprocessor.df = df_encoded
        df_normalized = preprocessor.normalize_features()
        assert abs(df_normalized[numeric_cols].mean().mean()) < 0.1, "Mean should be near 0"

        print("\n✅ Preprocessing Chain Test PASSED")

    def test_feature_engineering_comprehensive(self, sample_dataset):
        """Test comprehensive feature engineering.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        # Encode categorical first
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        df['department'] = encoder.fit_transform(df['department'])

        engineer = FeatureEngineer(df)

        # Test polynomial features
        df_poly = engineer.create_polynomial_features(['age'], degree=3)
        assert 'age_2' in df_poly.columns, "age_2 should exist"
        assert 'age_3' in df_poly.columns, "age_3 should exist"

        # Reset and test interaction features
        engineer.df = df
        df_inter = engineer.create_interaction_features(['age', 'income', 'experience'])
        assert 'age_x_income' in df_inter.columns, "age_x_income should exist"
        assert 'age_x_experience' in df_inter.columns, "age_x_experience should exist"
        assert 'income_x_experience' in df_inter.columns, "income_x_experience should exist"

        # Reset and test ratio features
        engineer.df = df
        df_ratio = engineer.create_ratio_features('income', 'age')
        assert 'income_div_age' in df_ratio.columns, "income_div_age should exist"

        print("\n✅ Feature Engineering Comprehensive Test PASSED")

    def test_data_splitting_stratification(self, sample_dataset):
        """Test data splitting with stratification.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        splitter = DataSplitter(df, target_col='burnout', test_size=0.2)

        # Test train/test split with stratification
        train, test = splitter.train_test_split(stratify=True)

        train_ratio = train['burnout'].mean()
        test_ratio = test['burnout'].mean()
        overall_ratio = df['burnout'].mean()

        # Stratification should keep ratios similar
        assert abs(train_ratio - overall_ratio) < 0.1, "Train ratio should match overall"
        assert abs(test_ratio - overall_ratio) < 0.1, "Test ratio should match overall"

        # Test train/val/test split
        train, val, test = splitter.train_val_test_split(
            train_size=0.6, val_size=0.2, test_size=0.2
        )

        assert len(train) == 120, "Train size should be 60%"
        assert len(val) == 40, "Val size should be 20%"
        assert len(test) == 40, "Test size should be 20%"

        print("\n✅ Data Splitting Stratification Test PASSED")

    def test_model_training_and_metrics(self, sample_dataset):
        """Test model training and metric calculation.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        # Preprocess
        numeric_cols = ['age', 'income', 'experience', 'hours_per_week']
        categorical_cols = ['department']

        preprocessor = DataPreprocessor(df, numeric_cols, categorical_cols)
        df = preprocessor.fit_transform()

        # Feature engineer
        engineer = FeatureEngineer(df)
        df = engineer.create_polynomial_features(['age'], degree=2)

        # Split
        splitter = DataSplitter(df, target_col='burnout', test_size=0.2)
        train, test = splitter.train_test_split(stratify=True)

        X_train = train.drop('burnout', axis=1).values
        y_train = train['burnout'].values
        X_test = test.drop('burnout', axis=1).values
        y_test = test['burnout'].values

        # Train
        trainer = ModelTrainer(cv=5)
        results = trainer.train_multiple(
            X_train, y_train,
            algorithms=['logistic_regression', 'random_forest'],
            params_dict={
                'logistic_regression': {'max_iter': 1000},
                'random_forest': {'n_estimators': 10}
            }
        )

        # Evaluate
        for result in results:
            predictions = result['model'].predict(X_test)
            evaluator = EvaluationMetrics(y_test, predictions)
            metrics = evaluator.calculate_metrics()

            assert 'accuracy' in metrics
            assert 'f1' in metrics
            assert isinstance(metrics['accuracy'], (int, float))
            assert 0 <= metrics['accuracy'] <= 1

        print("\n✅ Model Training and Metrics Test PASSED")

    def test_module_imports(self):
        """Test that all modules can be imported successfully.

        This is a smoke test to ensure all modules are available and
        can be imported without errors.
        """
        from src.data.loader import DataLoader  # noqa: F401
        from src.data.preprocessor import DataPreprocessor  # noqa: F401
        from src.data.splitter import DataSplitter  # noqa: F401
        from src.features.engineer import FeatureEngineer  # noqa: F401
        from src.models.trainer import ModelTrainer  # noqa: F401
        from src.models.base import BaseModel  # noqa: F401
        from src.evaluation.metrics import EvaluationMetrics  # noqa: F401
        from src.mlflow_integration.tracker import MLflowTracker  # noqa: F401
        from src.mlflow_integration.registry import MLflowRegistry  # noqa: F401
        from src.monitoring.logger import PredictionLogger  # noqa: F401
        from src.monitoring.performance import PerformanceTracker  # noqa: F401

        print("\n✅ All modules imported successfully")


class TestPipelineErrorHandling:
    """Test error handling in pipeline components."""

    def test_loader_file_not_found(self):
        """Test DataLoader raises error for non-existent file."""
        loader = DataLoader("/nonexistent/path/file.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_loader_empty_dataframe(self):
        """Test DataLoader raises error for empty CSV.

        Creates a temporary empty CSV and verifies appropriate error.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "empty.csv"
            csv_path.write_text("")

            loader = DataLoader(str(csv_path))

            with pytest.raises((ValueError, pd.errors.ParserError)):
                loader.load()

    def test_splitter_invalid_proportions(self, sample_dataset):
        """Test DataSplitter raises error for invalid proportions."""
        loader = DataLoader(sample_dataset)
        df = loader.load()

        splitter = DataSplitter(df, target_col='burnout')

        with pytest.raises(ValueError):
            splitter.train_val_test_split(
                train_size=0.5,
                val_size=0.3,
                test_size=0.3
            )

    def test_evaluation_proba_required_for_auc(self, sample_dataset):
        """Test EvaluationMetrics raises error when AUC calculated without probabilities."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])

        evaluator = EvaluationMetrics(y_true, y_pred, is_proba=False)

        with pytest.raises(ValueError):
            evaluator.calculate_auc()


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    def test_preprocessor_normalize_and_encode(self, sample_dataset):
        """Test that preprocessing preserves data integrity.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        original_rows = len(df)
        numeric_cols = ['age', 'income', 'experience', 'hours_per_week']
        categorical_cols = ['department']

        preprocessor = DataPreprocessor(df, numeric_cols, categorical_cols)
        df_processed = preprocessor.fit_transform()

        # Check row count preserved
        assert len(df_processed) == original_rows, "Row count should be preserved"

        # Check no data loss
        assert df_processed.shape[1] >= len(numeric_cols) + len(categorical_cols), \
            "All columns should be preserved"

    def test_feature_engineer_maintains_row_count(self, sample_dataset):
        """Test that feature engineering maintains row count.

        Args:
            sample_dataset: Path to sample dataset fixture
        """
        loader = DataLoader(sample_dataset)
        df = loader.load()

        original_rows = len(df)

        engineer = FeatureEngineer(df)
        df_poly = engineer.create_polynomial_features(['age'], degree=2)
        df_inter = engineer.create_interaction_features(['income', 'experience'])

        assert len(df_poly) == original_rows, "Row count should be preserved after polynomial features"
        assert len(df_inter) == original_rows, "Row count should be preserved after interaction features"
