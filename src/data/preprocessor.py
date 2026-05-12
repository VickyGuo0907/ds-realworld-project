"""Data preprocessing utilities."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Optional


class DataPreprocessor:
    """Preprocess data for modeling."""

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            df: DataFrame to preprocess
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
        """
        self.df = df.copy()
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.scaler = None
        self.encoders = {}

    def handle_missing_values(self, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            strategy: 'mean' for numeric, 'mode' for categorical

        Returns:
            DataFrame with missing values handled
        """
        df = self.df.copy()

        if strategy == "mean":
            for col in self.numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)

        return df

    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables.

        Returns:
            DataFrame with encoded categorical columns
        """
        df = self.df.copy()

        for col in self.categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self.encoders[col] = encoder

        return df

    def normalize_features(self) -> pd.DataFrame:
        """
        Normalize numeric features to zero mean, unit variance.

        Returns:
            DataFrame with normalized features
        """
        df = self.df.copy()

        self.scaler = StandardScaler()
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])

        return df

    def fit_transform(self, strategy: str = "mean") -> pd.DataFrame:
        """
        Apply all preprocessing steps.

        Args:
            strategy: Strategy for missing values

        Returns:
            Fully preprocessed DataFrame
        """
        df = self.handle_missing_values(strategy=strategy)
        self.df = df
        df = self.encode_categorical()
        self.df = df
        df = self.normalize_features()

        return df
