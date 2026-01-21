import os

import pandas as pd
import pytest

from vespatune import VespaTuneSplitter, split_data


class TestVespaTuneSplitter:
    def test_split_binary_classification(self, full_data_file):
        """Test splitting binary classification data with stratification."""
        splitter = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=5,
            targets=["target"],
            task="classification",
        )

        fold_files = splitter.split()

        assert len(fold_files) == 5
        for fold_info in fold_files:
            assert os.path.exists(fold_info["train_path"])
            assert os.path.exists(fold_info["valid_path"])
            assert fold_info["train_size"] + fold_info["valid_size"] == full_data_file["n_samples"]

    def test_split_regression(self, regression_data_file):
        """Test splitting regression data with binned stratification."""
        splitter = VespaTuneSplitter(
            input_filename=regression_data_file["data_path"],
            output_dir=os.path.join(regression_data_file["temp_dir"], "folds"),
            num_folds=3,
            targets=["target"],
            task="regression",
        )

        fold_files = splitter.split()

        assert len(fold_files) == 3
        for fold_info in fold_files:
            assert os.path.exists(fold_info["train_path"])
            assert os.path.exists(fold_info["valid_path"])

    def test_split_auto_detect_classification(self, full_data_file):
        """Test auto-detection of classification task."""
        splitter = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=5,
            targets=["target"],
            task=None,  # auto-detect
        )

        fold_files = splitter.split()

        assert len(fold_files) == 5

    def test_split_auto_detect_regression(self, regression_data_file):
        """Test auto-detection of regression task."""
        splitter = VespaTuneSplitter(
            input_filename=regression_data_file["data_path"],
            output_dir=os.path.join(regression_data_file["temp_dir"], "folds"),
            num_folds=5,
            targets=["target"],
            task=None,  # auto-detect
        )

        fold_files = splitter.split()

        assert len(fold_files) == 5

    def test_split_custom_num_folds(self, full_data_file):
        """Test custom number of folds."""
        splitter = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=10,
            targets=["target"],
        )

        fold_files = splitter.split()

        assert len(fold_files) == 10

    def test_split_data_function(self, full_data_file):
        """Test the split_data convenience function."""
        fold_files = split_data(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=5,
            targets=["target"],
        )

        assert len(fold_files) == 5

    def test_split_preserves_data(self, full_data_file):
        """Test that splitting preserves all rows without overlap."""
        splitter = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=5,
            targets=["target"],
        )

        fold_files = splitter.split()

        # Check that each valid fold has unique samples
        all_valid_ids = []
        for fold_info in fold_files:
            valid_df = pd.read_csv(fold_info["valid_path"])
            all_valid_ids.extend(valid_df["id"].tolist())

        # All IDs should be unique
        assert len(all_valid_ids) == len(set(all_valid_ids))
        # Should cover all samples
        assert len(all_valid_ids) == full_data_file["n_samples"]

    def test_split_with_seed_reproducibility(self, full_data_file):
        """Test that the same seed produces the same splits."""
        splitter1 = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds1"),
            num_folds=5,
            targets=["target"],
            seed=42,
        )
        fold_files1 = splitter1.split()

        splitter2 = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds2"),
            num_folds=5,
            targets=["target"],
            seed=42,
        )
        fold_files2 = splitter2.split()

        # Check that the same samples are in validation for each fold
        for f1, f2 in zip(fold_files1, fold_files2):
            valid_df1 = pd.read_csv(f1["valid_path"])
            valid_df2 = pd.read_csv(f2["valid_path"])
            assert valid_df1["id"].tolist() == valid_df2["id"].tolist()

    def test_split_default_target(self, full_data_file):
        """Test that default target column is used when not specified."""
        splitter = VespaTuneSplitter(
            input_filename=full_data_file["data_path"],
            output_dir=os.path.join(full_data_file["temp_dir"], "folds"),
            num_folds=5,
        )

        assert splitter.targets == ["target"]


class TestSplitterWithSampleData:
    """Tests using actual sample data files."""

    @pytest.fixture
    def binary_sample_file(self):
        return "data_samples/binary_classification.csv"

    @pytest.fixture
    def regression_sample_file(self):
        return "data_samples/single_column_regression.csv"

    def test_split_real_binary_data(self, binary_sample_file, temp_dir):
        """Test splitting real binary classification sample data."""
        if not os.path.exists(binary_sample_file):
            pytest.skip("Sample data not available")

        splitter = VespaTuneSplitter(
            input_filename=binary_sample_file,
            output_dir=os.path.join(temp_dir, "folds"),
            num_folds=5,
            targets=["income"],
            task="classification",
        )

        fold_files = splitter.split()

        assert len(fold_files) == 5
        total_samples = fold_files[0]["train_size"] + fold_files[0]["valid_size"]
        for fold_info in fold_files:
            assert fold_info["train_size"] + fold_info["valid_size"] == total_samples

    def test_split_real_regression_data(self, regression_sample_file, temp_dir):
        """Test splitting real regression sample data."""
        if not os.path.exists(regression_sample_file):
            pytest.skip("Sample data not available")

        splitter = VespaTuneSplitter(
            input_filename=regression_sample_file,
            output_dir=os.path.join(temp_dir, "folds"),
            num_folds=5,
            targets=["target"],
            task="regression",
        )

        fold_files = splitter.split()

        assert len(fold_files) == 5
