# Data Science Real-World Project

A personal learning project for data science experiments and prediction model development using real-world datasets from Kaggle.

## Overview

This repository contains a collection of data science projects focused on:
- Exploratory data analysis (EDA) on real-world datasets
- Building and evaluating prediction models
- Implementing machine learning solutions for various problem types
- Learning practical data science workflows and best practices

**Purpose**: Personal practice and learning in:
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Model evaluation and validation
- Production-ready code practices in ML

## Project Structure

```
ds_realworld_project/
├── README.md
├── pyproject.toml
├── main.py
├── data/
│   ├── raw/              # Original Kaggle datasets
│   └── processed/        # Cleaned and processed data
├── notebooks/            # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── data/            # Data loading and preprocessing modules
│   ├── features/        # Feature engineering utilities
│   ├── models/          # Model definitions and training logic
│   └── evaluation/      # Model evaluation and metrics
├── tests/               # Unit and integration tests
└── models/              # Trained model artifacts
```

## Getting Started

### Requirements
- Python 3.14+
- pip or poetry for dependency management
- Kaggle API credentials (optional, for automated downloads)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VickyGuo0907/ds-realworld-project.git
cd ds_realworld_project
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

### Download Datasets

#### Option A: Kaggle API (Automated)

1. Create Kaggle API credentials:
   - Visit https://www.kaggle.com/settings/account
   - Click "Create New API Token" (downloads `kaggle.json`)
   - Move to `~/.kaggle/`: `mv ~/Downloads/kaggle.json ~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. Download dataset:
```bash
python -m src.data.downloader <dataset-id>

# Example
python -m src.data.downloader rohitgajawada/developer-burnout
```

#### Option B: Manual Download
   - Visit [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Download your desired datasets
   - Extract to `data/raw/`

#### Kaggle API Complete Workflow

**1. Set up Kaggle API credentials:**
```bash
# Get API token from https://www.kaggle.com/settings/account
# Under "Legacy API Credentials" → "Create New API Token"
# This downloads kaggle.json

# Place credentials in home directory
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify installation
kaggle --version
```

**2. Find dataset ID:**

From Kaggle notebook or dataset page:
- Open the Kaggle notebook/dataset
- Look at the URL: `https://www.kaggle.com/datasets/{username}/{dataset-name}`
- Extract the ID: `username/dataset-name`

Or search via CLI:
```bash
kaggle datasets list --search "your-keyword"
```

**3. Download dataset:**
```bash
# Using project script
python -m src.data.downloader rohitgajawada/developer-burnout

# Or using Kaggle CLI directly
kaggle datasets download -d rohitgajawada/developer-burnout -p data/raw/
```

**4. Extract and verify:**
```bash
# Unzip downloaded file
unzip -q ~/Downloads/*.zip -d data/raw/

# List dataset files
ls data/raw/
ls -la data/raw/
```

**5. Load in notebook:**
```python
import pandas as pd

df = pd.read_csv('data/raw/your_dataset.csv')
df.info()
df.head()
```

**Security Note**: Never commit `kaggle.json` - it's in `.gitignore`

## Development Workflow

### Project Notebooks

Notebooks are stored in `notebooks/` directory:
- `01_developer_burnout_experiment.ipynb` - Main analysis from Kaggle

### Running Experiments

1. **Start Jupyter**:
```bash
source .venv/bin/activate
jupyter notebook notebooks/
```

2. **Load your data in notebook**:
```python
import pandas as pd
df = pd.read_csv('../data/raw/your_dataset.csv')
```

3. **Model Training**:
```bash
python src/models/train.py --dataset <dataset_name> --model <model_type>
```

4. **Evaluation**:
```bash
python src/evaluation/evaluate.py --model <model_path>
```

### Testing

Run tests with pytest:
```bash
pytest tests/ --cov=src
```

## Current Projects

Add your projects here as you complete them:

- [ ] Project 1: [Dataset Name] - [Problem Type]
- [ ] Project 2: [Dataset Name] - [Problem Type]

## Key Libraries

Dependencies will be added to `pyproject.toml` as projects develop:

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Modeling**: scikit-learn, xgboost, lightgbm, tensorflow/pytorch
- **ML Utilities**: scikit-learn, optuna
- **Testing**: pytest, pytest-cov

## Learning Goals

- [ ] Master data preprocessing and feature engineering
- [ ] Implement multiple model architectures
- [ ] Understand hyperparameter tuning strategies
- [ ] Practice proper model evaluation techniques
- [ ] Learn production ML best practices
- [ ] Document experiments and findings

## Best Practices

### Code Standards
- All functions have docstrings explaining purpose, parameters, and returns
- Type annotations on all function signatures
- Maximum function length of 40 lines
- Parameterized queries for any data access (no hardcoded values)
- Explicit error handling

### Data Management
- Raw data in `data/raw/` (never modified)
- Processed data in `data/processed/`
- Document data sources and preprocessing steps
- Use version control for datasets (DVC recommended for large files)

### Model Development
- Train/test split with proper stratification
- Cross-validation for robust evaluation
- Track hyperparameters and results
- Save model artifacts with metadata
- Document model limitations and assumptions

### Testing
- Unit tests for data processing functions
- Integration tests for complete pipelines
- Minimum 80% code coverage target
- Test edge cases and error conditions

## Resources

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [MLOps Best Practices](https://ml-eng.github.io/structuring-ml-projects/)

## Notes

This is a personal learning project. The focus is on:
1. Understanding machine learning fundamentals
2. Practicing with real-world, messy data
3. Developing reproducible workflows
4. Building production-quality code habits

All code follows PEP 8 standards with black formatting and ruff linting.

## License

Personal learning project - no specific license

## Author

Vicky Guo - Data Science Learning Project

---

**Started**: 2026-04-19

Happy learning! 🚀
