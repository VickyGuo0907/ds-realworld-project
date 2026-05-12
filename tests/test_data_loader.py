"""Tests for sources loader module."""

import pytest
import pandas as pd
from src.data_pipeline.loader import DataLoader


@pytest.fixture
def test_csv_file(tmp_path):
    """Create a test CSV file."""
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_csv(test_csv_file):
    """Test loading CSV file."""
    loader = DataLoader(test_csv_file)
    df = loader.load()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["feature1", "feature2", "target"]


def test_load_with_validation(test_csv_file):
    """Test loading with basic validation."""
    loader = DataLoader(test_csv_file)
    df = loader.load()

    assert df.isnull().sum().sum() == 0, "No null values expected in test sources"
    assert df.shape[0] > 0, "DataFrame should not be empty"


def test_load_nonexistent_file():
    """Test error handling for nonexistent file."""
    loader = DataLoader("nonexistent.csv")

    with pytest.raises(FileNotFoundError):
        loader.load()
