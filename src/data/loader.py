"""Data loading utilities."""

from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any


class DataLoader:
    """Load datasets from CSV files with validation."""

    def __init__(self, filepath: str) -> None:
        """
        Initialize DataLoader.

        Args:
            filepath: Path to CSV file
        """
        self.filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Returns:
            DataFrame with data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is empty or invalid format
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        if df.empty:
            raise ValueError("CSV file is empty")

        return df

    def load_with_info(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load CSV and return with metadata.

        Returns:
            Tuple of (DataFrame, dict with shape and columns info)
        """
        df = self.load()
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }
        return df, info
