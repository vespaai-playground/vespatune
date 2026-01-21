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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "cat_feature": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    })

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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "target": np.random.randint(0, 5, n_samples),
    })

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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 10 + 50,
    })

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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
        "target1": np.random.randn(n_samples) * 10,
        "target2": np.random.randn(n_samples) * 5,
    })

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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randint(0, 2, n_samples),
    })

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

    df = pd.DataFrame({
        "id": range(n_samples),
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "target": np.random.randn(n_samples) * 10 + 50,
    })

    data_path = os.path.join(temp_dir, "data.csv")
    df.to_csv(data_path, index=False)

    return {
        "data_path": data_path,
        "temp_dir": temp_dir,
        "n_samples": n_samples,
    }
