import pytest
import pandas as pd
from src.data.splitter import DataSplitter


@pytest.fixture
def sample_data():
    """Create sample data."""
    data = {"feature1": range(100), "feature2": range(100, 200), "target": [0, 1] * 50}
    return pd.DataFrame(data)


def test_train_test_split(sample_data):
    """Test train/test split."""
    splitter = DataSplitter(sample_data, target_col="target", test_size=0.2)
    train, test = splitter.train_test_split()

    assert len(train) + len(test) == len(sample_data)
    assert len(test) == int(len(sample_data) * 0.2)


def test_stratified_split(sample_data):
    """Test stratified split maintains class balance."""
    splitter = DataSplitter(sample_data, target_col="target", test_size=0.2)
    train, test = splitter.train_test_split(stratify=True)

    train_ratio = train["target"].value_counts(normalize=True).sort_index()
    test_ratio = test["target"].value_counts(normalize=True).sort_index()

    assert abs(train_ratio[0] - test_ratio[0]) < 0.05


def test_train_val_test_split(sample_data):
    """Test three-way split."""
    splitter = DataSplitter(sample_data, target_col="target")
    train, val, test = splitter.train_val_test_split(
        train_size=0.6, val_size=0.2, test_size=0.2
    )

    assert len(train) + len(val) + len(test) == len(sample_data)
    assert len(train) == int(len(sample_data) * 0.6)
