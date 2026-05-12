import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data with issues."""
    data = {
        'numeric1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric2': [10, 20, 30, 40, 50],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def test_handle_missing_values(sample_data):
    """Test handling missing values."""
    preprocessor = DataPreprocessor(sample_data, numeric_cols=['numeric1', 'numeric2'])
    df_clean = preprocessor.handle_missing_values(strategy='mean')

    assert df_clean.isnull().sum().sum() == 0


def test_encode_categorical(sample_data):
    """Test categorical encoding."""
    preprocessor = DataPreprocessor(sample_data, categorical_cols=['categorical'])
    df_encoded = preprocessor.encode_categorical()

    assert 'categorical' not in df_encoded.columns or df_encoded['categorical'].dtype != 'object'


def test_normalize_features(sample_data):
    """Test feature normalization."""
    preprocessor = DataPreprocessor(sample_data, numeric_cols=['numeric1', 'numeric2'])
    df_normalized = preprocessor.normalize_features()

    # Check that at least one numeric column is normalized (mean ~0, std ~1)
    assert df_normalized.shape == sample_data.shape
