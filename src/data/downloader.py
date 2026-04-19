"""Kaggle dataset downloader utilities."""

import subprocess
import os
from pathlib import Path


def download_kaggle_dataset(dataset_id: str, output_dir: str = "data/raw") -> None:
    """Download dataset from Kaggle using API.

    Args:
        dataset_id: Kaggle dataset identifier (e.g., 'username/dataset-name')
        output_dir: Directory to save downloaded files

    Raises:
        FileNotFoundError: If kaggle.json is not found
        subprocess.CalledProcessError: If download fails
    """
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_config.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_config}. "
            "Visit https://www.kaggle.com/settings/account to create API token."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_id} to {output_dir}...")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_id, "-p", output_dir],
        check=True
    )

    print("Download complete!")


def download_kaggle_notebook_dataset(
    notebook_path: str, output_dir: str = "data/raw"
) -> None:
    """Download dataset associated with a Kaggle notebook.

    Args:
        notebook_path: Kaggle notebook path (e.g., 'username/notebook-slug')
        output_dir: Directory to save downloaded files
    """
    print(f"Note: Notebook-associated datasets need manual identification.")
    print(f"Visit Kaggle notebook and check the 'Data' section for dataset IDs.")
    print(f"Then use: download_kaggle_dataset('dataset-id', '{output_dir}')")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python downloader.py <dataset-id>")
        print("Example: python downloader.py rohitgajawada/developer-burnout")
        sys.exit(1)

    dataset_id = sys.argv[1]
    download_kaggle_dataset(dataset_id)