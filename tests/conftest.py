import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def binary_classification_data(temp_dir):
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "cat_feature": np.random.choice(["A", "B", "C"], n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )

    train_df = df.iloc[:150].reset_index(drop=True)
    valid_df = df.iloc[150:180].reset_index(drop=True)
    test_df = df.iloc[180:].drop(columns=["target"]).reset_index(drop=True)

    train_path = os.path.join(temp_dir, "train.csv")
    valid_path = os.path.join(temp_dir, "valid.csv")
    test_path = os.path.join(temp_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "temp_dir": temp_dir,
        "targets": ["target"],
        "task": "classification",
    }


@pytest.fixture
def multiclass_classification_data(temp_dir):
    """Generate sample multi-class classification data."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "target": np.random.randint(0, 5, n_samples),
        }
    )

    train_df = df.iloc[:150].reset_index(drop=True)
    valid_df = df.iloc[150:180].reset_index(drop=True)
    test_df = df.iloc[180:].drop(columns=["target"]).reset_index(drop=True)

    train_path = os.path.join(temp_dir, "train.csv")
    valid_path = os.path.join(temp_dir, "valid.csv")
    test_path = os.path.join(temp_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "temp_dir": temp_dir,
        "targets": ["target"],
        "task": "classification",
    }


@pytest.fixture
def regression_data(temp_dir):
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "target": np.random.randn(n_samples) * 10 + 50,
        }
    )

    train_df = df.iloc[:150].reset_index(drop=True)
    valid_df = df.iloc[150:180].reset_index(drop=True)
    test_df = df.iloc[180:].drop(columns=["target"]).reset_index(drop=True)

    train_path = os.path.join(temp_dir, "train.csv")
    valid_path = os.path.join(temp_dir, "valid.csv")
    test_path = os.path.join(temp_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "temp_dir": temp_dir,
        "targets": ["target"],
        "task": "regression",
    }


@pytest.fixture
def multi_target_regression_data(temp_dir):
    """Generate sample multi-target regression data."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "target1": np.random.randn(n_samples) * 10,
            "target2": np.random.randn(n_samples) * 5,
        }
    )

    train_df = df.iloc[:150].reset_index(drop=True)
    valid_df = df.iloc[150:180].reset_index(drop=True)
    test_df = df.iloc[180:].drop(columns=["target1", "target2"]).reset_index(drop=True)

    train_path = os.path.join(temp_dir, "train.csv")
    valid_path = os.path.join(temp_dir, "valid.csv")
    test_path = os.path.join(temp_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "temp_dir": temp_dir,
        "targets": ["target1", "target2"],
        "task": "regression",
    }


@pytest.fixture
def full_data_file(temp_dir):
    """Generate a single data file for splitter testing."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }
    )

    data_path = os.path.join(temp_dir, "data.csv")
    df.to_csv(data_path, index=False)

    return {
        "data_path": data_path,
        "temp_dir": temp_dir,
        "n_samples": n_samples,
    }


@pytest.fixture
def regression_data_file(temp_dir):
    """Generate a single regression data file for splitter testing."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "id": range(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "target": np.random.randn(n_samples) * 10 + 50,
        }
    )

    data_path = os.path.join(temp_dir, "data.csv")
    df.to_csv(data_path, index=False)

    return {
        "data_path": data_path,
        "temp_dir": temp_dir,
        "n_samples": n_samples,
    }


# ============================================================================
# Real data fixtures using data_samples
# ============================================================================

DATA_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data_samples")


@pytest.fixture
def real_binary_classification_data(temp_dir):
    """Load real binary classification data from data_samples."""
    train_path = os.path.join(DATA_SAMPLES_DIR, "binary_classification", "train_fold_0.csv")
    valid_path = os.path.join(DATA_SAMPLES_DIR, "binary_classification", "valid_fold_0.csv")

    if not os.path.exists(train_path):
        pytest.skip("Binary classification sample data not available")

    # Load only first 100 rows for faster tests
    train_df = pd.read_csv(train_path, nrows=100)
    valid_df = pd.read_csv(valid_path, nrows=30)

    # Add id column if not present
    if "id" not in train_df.columns:
        train_df.insert(0, "id", range(len(train_df)))
        valid_df.insert(0, "id", range(len(train_df), len(train_df) + len(valid_df)))

    # Save to temp dir
    train_out = os.path.join(temp_dir, "train.csv")
    valid_out = os.path.join(temp_dir, "valid.csv")
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)

    # Identify features and target
    target_col = "income"
    feature_cols = [c for c in train_df.columns if c not in ["id", target_col]]

    return {
        "train_path": train_out,
        "valid_path": valid_out,
        "temp_dir": temp_dir,
        "targets": [target_col],
        "features": feature_cols,
        "task": "classification",
        "train_df": train_df,
        "valid_df": valid_df,
    }


@pytest.fixture
def real_multi_class_classification_data(temp_dir):
    """Load real multi-class classification data from data_samples (Iris dataset)."""
    train_path = os.path.join(DATA_SAMPLES_DIR, "multi_class_classification", "train_fold_0.csv")
    valid_path = os.path.join(DATA_SAMPLES_DIR, "multi_class_classification", "valid_fold_0.csv")

    if not os.path.exists(train_path):
        pytest.skip("Multi-class classification sample data not available")

    train_df = pd.read_csv(train_path, nrows=100)
    valid_df = pd.read_csv(valid_path, nrows=30)

    train_out = os.path.join(temp_dir, "train.csv")
    valid_out = os.path.join(temp_dir, "valid.csv")
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)

    target_col = "target"
    feature_cols = [c for c in train_df.columns if c not in ["id", target_col]]

    return {
        "train_path": train_out,
        "valid_path": valid_out,
        "temp_dir": temp_dir,
        "targets": [target_col],
        "features": feature_cols,
        "task": "classification",
        "train_df": train_df,
        "valid_df": valid_df,
    }


@pytest.fixture
def real_single_column_regression_data(temp_dir):
    """Load real single column regression data from data_samples."""
    train_path = os.path.join(DATA_SAMPLES_DIR, "single_column_regression", "train_fold_0.csv")
    valid_path = os.path.join(DATA_SAMPLES_DIR, "single_column_regression", "valid_fold_0.csv")

    if not os.path.exists(train_path):
        pytest.skip("Single column regression sample data not available")

    train_df = pd.read_csv(train_path, nrows=100)
    valid_df = pd.read_csv(valid_path, nrows=30)

    train_out = os.path.join(temp_dir, "train.csv")
    valid_out = os.path.join(temp_dir, "valid.csv")
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)

    target_col = "target"
    feature_cols = [c for c in train_df.columns if c not in ["id", target_col]]

    return {
        "train_path": train_out,
        "valid_path": valid_out,
        "temp_dir": temp_dir,
        "targets": [target_col],
        "features": feature_cols,
        "task": "regression",
        "train_df": train_df,
        "valid_df": valid_df,
    }


@pytest.fixture
def real_multi_column_regression_data(temp_dir):
    """Load real multi-column regression data from data_samples."""
    train_path = os.path.join(DATA_SAMPLES_DIR, "multi_column_regression", "train_fold_0.csv")
    valid_path = os.path.join(DATA_SAMPLES_DIR, "multi_column_regression", "valid_fold_0.csv")

    if not os.path.exists(train_path):
        pytest.skip("Multi-column regression sample data not available")

    train_df = pd.read_csv(train_path, nrows=100)
    valid_df = pd.read_csv(valid_path, nrows=30)

    train_out = os.path.join(temp_dir, "train.csv")
    valid_out = os.path.join(temp_dir, "valid.csv")
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)

    target_cols = ["target1", "target2", "target3"]
    feature_cols = [c for c in train_df.columns if c not in ["id"] + target_cols]

    return {
        "train_path": train_out,
        "valid_path": valid_out,
        "temp_dir": temp_dir,
        "targets": target_cols,
        "features": feature_cols,
        "task": "regression",
        "train_df": train_df,
        "valid_df": valid_df,
    }


@pytest.fixture
def real_multi_label_classification_data(temp_dir):
    """Load real multi-label classification data from data_samples."""
    train_path = os.path.join(DATA_SAMPLES_DIR, "multi_label_classification", "train_fold_0.csv")
    valid_path = os.path.join(DATA_SAMPLES_DIR, "multi_label_classification", "valid_fold_0.csv")

    if not os.path.exists(train_path):
        pytest.skip("Multi-label classification sample data not available")

    train_df = pd.read_csv(train_path, nrows=100)
    valid_df = pd.read_csv(valid_path, nrows=30)

    train_out = os.path.join(temp_dir, "train.csv")
    valid_out = os.path.join(temp_dir, "valid.csv")
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)

    target_cols = ["service_a", "service_b"]
    feature_cols = [c for c in train_df.columns if c not in ["id"] + target_cols]

    return {
        "train_path": train_out,
        "valid_path": valid_out,
        "temp_dir": temp_dir,
        "targets": target_cols,
        "features": feature_cols,
        "task": "classification",
        "train_df": train_df,
        "valid_df": valid_df,
    }
