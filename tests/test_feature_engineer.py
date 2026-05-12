import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'age': [20, 25, 30, 35, 40],
        'income': [30000, 40000, 50000, 60000, 70000],
        'experience_years': [1, 2, 5, 8, 10],
        'target': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def test_create_polynomial_features(sample_data):
    """Test polynomial feature creation."""
    engineer = FeatureEngineer(sample_data)
    df = engineer.create_polynomial_features(cols=['age'], degree=2)

    assert 'age_2' in df.columns
    assert df['age_2'].iloc[0] == 400  # 20^2


def test_create_interaction_features(sample_data):
    """Test interaction feature creation."""
    engineer = FeatureEngineer(sample_data)
    df = engineer.create_interaction_features(cols=['age', 'income'])

    assert 'age_x_income' in df.columns
    assert df['age_x_income'].iloc[0] == 600000  # 20 * 30000


def test_create_ratio_features(sample_data):
    """Test ratio feature creation."""
    engineer = FeatureEngineer(sample_data)
    df = engineer.create_ratio_features(
        numerator='income',
        denominator='experience_years'
    )

    assert 'income_div_experience_years' in df.columns
    assert df['income_div_experience_years'].iloc[0] == 30000.0  # 30000/1
