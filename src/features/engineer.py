"""Feature engineering utilities."""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Create and transform features."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize feature engineer.

        Args:
            df: DataFrame to engineer features on
        """
        self.df = df.copy()

    def create_polynomial_features(
        self, cols: List[str], degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features.

        Args:
            cols: Column names to create polynomials for
            degree: Polynomial degree

        Returns:
            DataFrame with new polynomial features
        """
        df = self.df.copy()

        for col in cols:
            for d in range(2, degree + 1):
                df[f"{col}_{d}"] = df[col] ** d

        return df

    def create_interaction_features(self, cols: List[str]) -> pd.DataFrame:
        """
        Create interaction features between columns.

        Args:
            cols: Column names to interact

        Returns:
            DataFrame with interaction features
        """
        df = self.df.copy()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        return df

    def create_ratio_features(self, numerator: str, denominator: str) -> pd.DataFrame:
        """
        Create ratio feature.

        Args:
            numerator: Numerator column
            denominator: Denominator column

        Returns:
            DataFrame with ratio feature
        """
        df = self.df.copy()

        # Avoid division by zero
        df[f"{numerator}_div_{denominator}"] = np.where(
            df[denominator] != 0, df[numerator] / df[denominator], 0
        )

        return df
