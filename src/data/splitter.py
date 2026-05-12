"""Data splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataSplitter:
    """Split data into train/test/validation sets."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Initialize splitter.

        Args:
            df: DataFrame to split
            target_col: Name of target column
            test_size: Proportion for test set
            random_state: Random seed
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def train_test_split(self, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into train and test sets.

        Args:
            stratify: Whether to stratify by target

        Returns:
            Tuple of (train_df, test_df)
        """
        stratify_col = self.df[self.target_col] if stratify else None

        train, test = train_test_split(
            self.df,
            test_size=self.test_size,
            stratify=stratify_col,
            random_state=self.random_state
        )

        return train.reset_index(drop=True), test.reset_index(drop=True)

    def train_val_test_split(
        self,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train, validation, and test sets.

        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            stratify: Whether to stratify by target

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not (abs(train_size + val_size + test_size - 1.0) < 0.01):
            raise ValueError("Proportions must sum to 1.0")

        stratify_col = self.df[self.target_col] if stratify else None

        # First split: train vs (val + test)
        train, temp = train_test_split(
            self.df,
            train_size=train_size,
            stratify=stratify_col,
            random_state=self.random_state
        )

        # Second split: split temp into val and test
        val_test_ratio = val_size / (val_size + test_size)
        stratify_col_temp = temp[self.target_col] if stratify else None

        val, test = train_test_split(
            temp,
            train_size=val_test_ratio,
            stratify=stratify_col_temp,
            random_state=self.random_state
        )

        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True)
        )
