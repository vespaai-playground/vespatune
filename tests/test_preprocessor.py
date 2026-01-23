import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from vespatune import (
    BasePreprocessor,
    CatBoostPreprocessor,
    LightGBMPreprocessor,
    XGBoostPreprocessor,
    get_preprocessor,
)
from vespatune.enums import ProblemType


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "id": range(n_samples),
            "cat1": np.random.choice(["A", "B", "C"], n_samples),
            "cat2": np.random.choice(["X", "Y"], n_samples),
            "num1": np.random.randn(n_samples),
            "num2": np.random.randn(n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }
    )


class TestGetPreprocessor:
    """Test the get_preprocessor factory function."""

    def test_get_xgboost_preprocessor(self):
        preprocessor = get_preprocessor("xgboost")
        assert isinstance(preprocessor, XGBoostPreprocessor)

    def test_get_lightgbm_preprocessor(self):
        preprocessor = get_preprocessor("lightgbm")
        assert isinstance(preprocessor, LightGBMPreprocessor)

    def test_get_catboost_preprocessor(self):
        preprocessor = get_preprocessor("catboost")
        assert isinstance(preprocessor, CatBoostPreprocessor)

    def test_get_preprocessor_with_params(self):
        preprocessor = get_preprocessor(
            "xgboost",
            features=["a", "b"],
            categorical_features=["a"],
        )
        assert preprocessor.features == ["a", "b"]
        assert preprocessor.categorical_features == ["a"]

    def test_get_preprocessor_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_preprocessor("unknown_model")


class TestXGBoostPreprocessor:
    """Test XGBoostPreprocessor."""

    def test_fit_and_transform(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        assert preprocessor._is_fitted
        result = preprocessor.transform(sample_data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(sample_data), 4)  # 2 cat + 2 num features

    def test_categorical_encoding(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)

        # Categorical features should be encoded as floats
        assert not np.any(np.isnan(result[:, :2]))  # No NaN in known data

    def test_handles_unknown_categories(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        test_data = pd.DataFrame(
            {
                "id": [0],
                "cat1": ["UNKNOWN"],
                "cat2": ["UNKNOWN"],
                "num1": [0.5],
                "num2": [0.5],
                "target": [0],
            }
        )

        result = preprocessor.transform(test_data)
        # Unknown categories should be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])


class TestLightGBMPreprocessor:
    """Test LightGBMPreprocessor."""

    def test_fit_and_transform(self, sample_data):
        preprocessor = LightGBMPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        assert preprocessor._is_fitted
        result = preprocessor.transform(sample_data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(sample_data), 4)

    def test_get_categorical_indices(self, sample_data):
        preprocessor = LightGBMPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        indices = preprocessor.get_categorical_indices()
        assert len(indices) == 2  # cat1 and cat2


class TestCatBoostPreprocessor:
    """Test CatBoostPreprocessor."""

    def test_fit_and_transform(self, sample_data):
        preprocessor = CatBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        assert preprocessor._is_fitted
        result = preprocessor.transform(sample_data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (len(sample_data), 4)

    def test_unknown_categories_mapped_to_minus_one(self, sample_data):
        preprocessor = CatBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        test_data = pd.DataFrame(
            {
                "id": [0],
                "cat1": ["UNKNOWN"],
                "cat2": ["UNKNOWN"],
                "num1": [0.5],
                "num2": [0.5],
                "target": [0],
            }
        )

        result = preprocessor.transform(test_data)
        # CatBoost uses -1 for unknown/missing
        assert result[0, 0] == -1
        assert result[0, 1] == -1


class TestPreprocessorTargetEncoding:
    """Test target encoding across preprocessors."""

    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "num1": np.random.randn(n_samples),
                "target": np.random.choice(["yes", "no"], n_samples),
            }
        )

    @pytest.mark.parametrize(
        "preprocessor_class",
        [
            XGBoostPreprocessor,
            LightGBMPreprocessor,
            CatBoostPreprocessor,
        ],
    )
    def test_target_encoding(self, classification_data, preprocessor_class):
        preprocessor = preprocessor_class()
        preprocessor.fit(
            classification_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        y_transformed = preprocessor.transform_target(classification_data["target"])
        assert set(np.unique(y_transformed)) == {0, 1}

        y_original = preprocessor.inverse_transform_target(y_transformed)
        assert set(y_original) == {"yes", "no"}

    @pytest.mark.parametrize(
        "preprocessor_class",
        [
            XGBoostPreprocessor,
            LightGBMPreprocessor,
            CatBoostPreprocessor,
        ],
    )
    def test_no_target_encoding_for_regression(self, sample_data, preprocessor_class):
        sample_data["target"] = np.random.randn(len(sample_data))
        preprocessor = preprocessor_class()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.single_column_regression,
            targets=["target"],
            idx="id",
        )

        assert preprocessor.target_encoder_ is None


class TestPreprocessorSaveLoad:
    """Test preprocessor serialization."""

    @pytest.mark.parametrize(
        "preprocessor_class",
        [
            XGBoostPreprocessor,
            LightGBMPreprocessor,
            CatBoostPreprocessor,
        ],
    )
    def test_save_load(self, sample_data, preprocessor_class):
        preprocessor = preprocessor_class()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "preprocessor")
            preprocessor.save(path)

            loaded = BasePreprocessor.load(path)

            assert loaded._is_fitted
            assert type(loaded) is type(preprocessor)
            assert loaded.features_ == preprocessor.features_

            # Test that loaded preprocessor produces same output
            original_output = preprocessor.transform(sample_data)
            loaded_output = loaded.transform(sample_data)
            np.testing.assert_array_equal(original_output, loaded_output)


class TestPreprocessorHelpers:
    """Test preprocessor helper methods."""

    def test_get_feature_names_out(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        names = preprocessor.get_feature_names_out()
        assert "cat1" in names
        assert "cat2" in names
        assert "num1" in names
        assert "num2" in names
        assert len(names) == 4

    def test_get_config(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        config = preprocessor.get_config()
        assert "features" in config
        assert "categorical_features" in config
        assert "n_features_in" in config
        assert config["has_target_encoder"] is True

    def test_repr(self, sample_data):
        preprocessor = XGBoostPreprocessor()
        assert "fitted=False" in repr(preprocessor)

        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )
        assert "fitted=True" in repr(preprocessor)
        assert "XGBoostPreprocessor" in repr(preprocessor)


class TestPreprocessorEncoding:
    """Test preprocessor encoding options."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "cat1": np.random.choice(["A", "B", "C"], n_samples),
                "cat2": np.random.choice(["X", "Y"], n_samples),
                "num1": np.random.randn(n_samples),
                "target": np.random.choice([0, 1], n_samples),
            }
        )

    def test_ordinal_encoding_default(self, sample_data):
        """Test that ordinal encoding is the default."""
        preprocessor = XGBoostPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)
        # Ordinal encoding: 2 cat features + 1 num feature = 3 columns
        assert result.shape == (len(sample_data), 3)
        assert preprocessor.encoding == "ordinal"

    def test_onehot_encoding(self, sample_data):
        """Test one-hot encoding expands categorical columns."""
        from vespatune.models.logreg_model import LogRegPreprocessor

        # Use LogRegPreprocessor which has encoding="onehot" as class attribute
        preprocessor = LogRegPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)
        # One-hot encoding: cat1 has 3 values, cat2 has 2 values, + 1 num = 6 columns
        assert result.shape == (len(sample_data), 6)

    def test_onehot_unknown_categories(self, sample_data):
        """Test that unknown categories become all zeros with one-hot encoding."""
        from vespatune.models.logreg_model import LogRegPreprocessor

        preprocessor = LogRegPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        test_data = pd.DataFrame(
            {
                "id": [0],
                "cat1": ["UNKNOWN"],
                "cat2": ["UNKNOWN"],
                "num1": [0.5],
                "target": [0],
            }
        )

        result = preprocessor.transform(test_data)
        # Unknown categories should be all zeros in one-hot encoding
        # First 3 columns are cat1 (A, B, C), next 2 are cat2 (X, Y)
        assert result[0, 0] == 0  # cat1_A
        assert result[0, 1] == 0  # cat1_B
        assert result[0, 2] == 0  # cat1_C
        assert result[0, 3] == 0  # cat2_X
        assert result[0, 4] == 0  # cat2_Y

    def test_logreg_preprocessor_uses_onehot(self, sample_data):
        """Test that LogRegPreprocessor always uses one-hot encoding (class attribute)."""
        from vespatune.models.logreg_model import LogRegPreprocessor

        # encoding is a class attribute, not configurable
        assert LogRegPreprocessor.encoding == "onehot"

        preprocessor = LogRegPreprocessor()
        assert preprocessor.encoding == "onehot"

        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)
        # One-hot: cat1(3) + cat2(2) + num1(1) = 6 columns
        assert result.shape == (len(sample_data), 6)

    def test_get_categorical_indices_onehot(self, sample_data):
        """Test get_categorical_indices returns all one-hot columns."""
        from vespatune.models.logreg_model import LogRegPreprocessor

        preprocessor = LogRegPreprocessor()
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        indices = preprocessor.get_categorical_indices()
        # Should include all 5 one-hot encoded columns (3 for cat1, 2 for cat2)
        assert len(indices) == 5


class TestPreprocessorErrors:
    """Test preprocessor error handling."""

    def test_transform_before_fit(self):
        preprocessor = XGBoostPreprocessor()
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(df)

    def test_save_before_fit(self):
        preprocessor = XGBoostPreprocessor()

        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.save("/tmp/preprocessor")


class TestPreprocessorImputation:
    """Test preprocessor imputation functionality."""

    @pytest.fixture
    def data_with_missing(self):
        """Create sample data with missing values."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "id": range(n_samples),
                "cat1": np.random.choice(["A", "B", "C", None], n_samples),
                "num1": np.random.randn(n_samples),
                "num2": np.random.randn(n_samples),
                "target": np.random.choice([0, 1], n_samples),
            }
        )
        # Add some NaN values to numeric columns
        df.loc[0:10, "num1"] = np.nan
        df.loc[5:15, "num2"] = np.nan
        return df

    def test_numeric_imputation_median(self, data_with_missing):
        preprocessor = XGBoostPreprocessor(
            numeric_impute_strategy="median",
            categorical_impute_strategy="most_frequent",  # Also impute categorical NaN
        )
        preprocessor.fit(
            data_with_missing,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(data_with_missing)

        # No NaN values should remain in output
        assert not np.any(np.isnan(result))

    def test_numeric_imputation_mean(self, data_with_missing):
        preprocessor = XGBoostPreprocessor(
            numeric_impute_strategy="mean",
            categorical_impute_strategy="most_frequent",  # Also impute categorical NaN
        )
        preprocessor.fit(
            data_with_missing,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(data_with_missing)
        assert not np.any(np.isnan(result))

    def test_categorical_imputation(self, data_with_missing):
        preprocessor = XGBoostPreprocessor(categorical_impute_strategy="most_frequent")
        preprocessor.fit(
            data_with_missing,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(data_with_missing)
        # Should have valid encoded values (no NaN from missing categories)
        assert result.shape[0] == len(data_with_missing)


class TestPreprocessorScaling:
    """Test preprocessor scaling functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "cat1": np.random.choice(["A", "B", "C"], n_samples),
                "num1": np.random.randn(n_samples) * 100 + 500,  # Large values
                "num2": np.random.randn(n_samples) * 0.01,  # Small values
                "target": np.random.choice([0, 1], n_samples),
            }
        )

    def test_standard_scaler(self, sample_data):
        preprocessor = XGBoostPreprocessor(scaler="standard")
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)

        # Get numeric column indices (after categorical)
        num_indices = preprocessor.get_categorical_indices()
        num_start = len(num_indices) if num_indices else 0

        # Scaled numeric features should have mean ~0 and std ~1
        numeric_data = result[:, num_start:]
        assert np.abs(numeric_data.mean()) < 0.1
        assert np.abs(numeric_data.std() - 1.0) < 0.1

    def test_minmax_scaler(self, sample_data):
        preprocessor = XGBoostPreprocessor(scaler="minmax")
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)

        # Get numeric columns
        num_indices = preprocessor.get_categorical_indices()
        num_start = len(num_indices) if num_indices else 0
        numeric_data = result[:, num_start:]

        # MinMax scaled data should be in [0, 1]
        assert numeric_data.min() >= -0.01  # Small tolerance
        assert numeric_data.max() <= 1.01

    def test_robust_scaler(self, sample_data):
        preprocessor = XGBoostPreprocessor(scaler="robust")
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(sample_data)
        assert result.shape == (len(sample_data), 3)  # 1 cat + 2 num

    def test_combined_imputation_and_scaling(self):
        """Test that imputation and scaling work together correctly."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "id": range(100),
                "num1": np.random.randn(100) * 100,
                "num2": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        df.loc[0:10, "num1"] = np.nan

        preprocessor = XGBoostPreprocessor(
            numeric_impute_strategy="median",
            scaler="standard",
        )
        preprocessor.fit(
            df,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        result = preprocessor.transform(df)

        # No NaN and approximately standardized
        assert not np.any(np.isnan(result))
        assert np.abs(result.mean()) < 0.1

    def test_config_includes_preprocessing_params(self, sample_data):
        """Test that get_config includes preprocessing parameters."""
        preprocessor = XGBoostPreprocessor(
            numeric_impute_strategy="median",
            categorical_impute_strategy="most_frequent",
            scaler="standard",
        )
        preprocessor.fit(
            sample_data,
            problem_type=ProblemType.binary_classification,
            targets=["target"],
            idx="id",
        )

        config = preprocessor.get_config()

        assert config["numeric_impute_strategy"] == "median"
        assert config["categorical_impute_strategy"] == "most_frequent"
        assert config["scaler"] == "standard"
