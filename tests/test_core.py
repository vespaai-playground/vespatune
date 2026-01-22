import os

import joblib
import pytest

from vespatune import VespaTune
from vespatune.enums import ProblemType


class TestVespaTuneInit:
    def test_init_binary_classification(self, binary_classification_data):
        """Test VespaTune initialization for binary classification."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        assert vtune.train_filename == data["train_path"]
        assert vtune.valid_filename == data["valid_path"]
        assert vtune.targets == ["target"]
        assert os.path.exists(output_dir)

    def test_init_regression(self, regression_data):
        """Test VespaTune initialization for regression."""
        data = regression_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        assert vtune.task == "regression"
        assert os.path.exists(output_dir)

    def test_init_default_targets(self, binary_classification_data):
        """Test default target column."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            num_trials=1,
        )

        assert vtune.targets == ["target"]

    def test_init_default_idx(self, binary_classification_data):
        """Test default index column."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            num_trials=1,
        )

        assert vtune.idx == "id"


class TestVespaTuneProblemTypeDetection:
    def test_auto_detect_binary_classification(self, binary_classification_data):
        """Test auto-detection of binary classification."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=None,  # Auto-detect
            num_trials=1,
        )

        # Process data to trigger problem type detection
        vtune._process_data()

        assert vtune.model_config.problem_type == ProblemType.binary_classification

    def test_auto_detect_multiclass_classification(self, multiclass_classification_data):
        """Test auto-detection of multiclass classification."""
        data = multiclass_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=None,  # Auto-detect
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.problem_type == ProblemType.multi_class_classification

    def test_auto_detect_regression(self, regression_data):
        """Test auto-detection of regression."""
        data = regression_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=None,  # Auto-detect
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.problem_type == ProblemType.single_column_regression

    def test_explicit_task_classification(self, binary_classification_data):
        """Test explicit classification task."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task="classification",
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.problem_type == ProblemType.binary_classification

    def test_explicit_task_regression(self, regression_data):
        """Test explicit regression task."""
        data = regression_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task="regression",
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.problem_type == ProblemType.single_column_regression


class TestVespaTuneDataProcessing:
    def test_process_data_creates_feather_files(self, binary_classification_data):
        """Test that data processing creates feather files."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        assert os.path.exists(os.path.join(output_dir, "train.feather"))
        assert os.path.exists(os.path.join(output_dir, "valid.feather"))

    def test_process_data_saves_config(self, binary_classification_data):
        """Test that data processing saves model config."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        config_path = os.path.join(output_dir, "vtune.config")
        assert os.path.exists(config_path)

        config = joblib.load(config_path)
        assert config.targets == data["targets"]

    def test_process_data_saves_encoders(self, binary_classification_data):
        """Test that data processing saves encoders."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        assert os.path.exists(os.path.join(output_dir, "vtune.categorical_encoder"))
        assert os.path.exists(os.path.join(output_dir, "vtune.target_encoder"))

    def test_categorical_encoding(self, binary_classification_data):
        """Test categorical feature encoding."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        # Check that categorical encoder was created
        encoder = joblib.load(os.path.join(output_dir, "vtune.categorical_encoder"))
        assert encoder is not None

        # Check that categorical features are in config
        assert "cat_feature" in vtune.model_config.categorical_features

    def test_process_with_test_file(self, binary_classification_data):
        """Test data processing with test file."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            test_filename=data["test_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        assert os.path.exists(os.path.join(output_dir, "test.feather"))


class TestVespaTuneTraining:
    @pytest.mark.slow
    def test_train_binary_classification(self, binary_classification_data):
        """Test full training pipeline for binary classification."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=2,  # Minimal trials for testing
            time_limit=60,
        )

        vtune.train()

        # Check that model was saved
        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))
        assert os.path.exists(os.path.join(output_dir, "vtune.best_params"))

    @pytest.mark.slow
    def test_train_regression(self, regression_data):
        """Test full training pipeline for regression."""
        data = regression_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_multi_target_regression(self, multi_target_regression_data):
        """Test training with multiple targets."""
        data = multi_target_regression_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        # For multi-target, model should be a list
        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])


class TestVespaTuneFeatures:
    def test_custom_features(self, binary_classification_data):
        """Test specifying custom features."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        custom_features = ["feature1", "feature2"]

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            features=custom_features,
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.features == custom_features

    def test_custom_categorical_features(self, binary_classification_data):
        """Test specifying custom categorical features."""
        data = binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            categorical_features=["cat_feature"],
            task=data["task"],
            num_trials=1,
        )

        vtune._process_data()

        assert vtune.model_config.categorical_features == ["cat_feature"]


# ============================================================================
# Real Data Tests - VespaTune Integration
# ============================================================================


class TestVespaTuneRealDataBinaryClassification:
    """Test VespaTune with real binary classification data."""

    @pytest.mark.slow
    def test_train_xgboost_real_data(self, real_binary_classification_data):
        """Test full training pipeline with XGBoost on real binary classification data."""
        data = real_binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_xgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="xgboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))
        assert os.path.exists(os.path.join(output_dir, "vtune.best_params"))

    @pytest.mark.slow
    def test_train_lightgbm_real_data(self, real_binary_classification_data):
        """Test full training pipeline with LightGBM on real binary classification data."""
        data = real_binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_lgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="lightgbm",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_catboost_real_data(self, real_binary_classification_data):
        """Test full training pipeline with CatBoost on real binary classification data."""
        data = real_binary_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_cb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="catboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))


class TestVespaTuneRealDataMultiClassClassification:
    """Test VespaTune with real multi-class classification data."""

    @pytest.mark.slow
    def test_train_xgboost_multiclass_real_data(self, real_multi_class_classification_data):
        """Test full training pipeline with XGBoost on real multi-class data."""
        data = real_multi_class_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_xgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="xgboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_lightgbm_multiclass_real_data(self, real_multi_class_classification_data):
        """Test full training pipeline with LightGBM on real multi-class data."""
        data = real_multi_class_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_lgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="lightgbm",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_catboost_multiclass_real_data(self, real_multi_class_classification_data):
        """Test full training pipeline with CatBoost on real multi-class data."""
        data = real_multi_class_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_cb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="catboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))


class TestVespaTuneRealDataRegression:
    """Test VespaTune with real regression data."""

    @pytest.mark.slow
    def test_train_xgboost_regression_real_data(self, real_single_column_regression_data):
        """Test full training pipeline with XGBoost on real regression data."""
        data = real_single_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_xgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="xgboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_lightgbm_regression_real_data(self, real_single_column_regression_data):
        """Test full training pipeline with LightGBM on real regression data."""
        data = real_single_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_lgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="lightgbm",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))

    @pytest.mark.slow
    def test_train_catboost_regression_real_data(self, real_single_column_regression_data):
        """Test full training pipeline with CatBoost on real regression data."""
        data = real_single_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_cb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="catboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        assert os.path.exists(os.path.join(output_dir, "vtune_model.final"))


class TestVespaTuneRealDataMultiColumnRegression:
    """Test VespaTune with real multi-column regression data."""

    @pytest.mark.slow
    def test_train_xgboost_multi_regression_real_data(self, real_multi_column_regression_data):
        """Test full training pipeline with XGBoost on real multi-target regression data."""
        data = real_multi_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_xgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="xgboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])

    @pytest.mark.slow
    def test_train_lightgbm_multi_regression_real_data(self, real_multi_column_regression_data):
        """Test full training pipeline with LightGBM on real multi-target regression data."""
        data = real_multi_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_lgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="lightgbm",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])

    @pytest.mark.slow
    def test_train_catboost_multi_regression_real_data(self, real_multi_column_regression_data):
        """Test full training pipeline with CatBoost on real multi-target regression data."""
        data = real_multi_column_regression_data
        output_dir = os.path.join(data["temp_dir"], "output_cb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="catboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])


class TestVespaTuneRealDataMultiLabelClassification:
    """Test VespaTune with real multi-label classification data."""

    @pytest.mark.slow
    def test_train_xgboost_multilabel_real_data(self, real_multi_label_classification_data):
        """Test full training pipeline with XGBoost on real multi-label data."""
        data = real_multi_label_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_xgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="xgboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])

    @pytest.mark.slow
    def test_train_lightgbm_multilabel_real_data(self, real_multi_label_classification_data):
        """Test full training pipeline with LightGBM on real multi-label data."""
        data = real_multi_label_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_lgb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="lightgbm",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])

    @pytest.mark.slow
    def test_train_catboost_multilabel_real_data(self, real_multi_label_classification_data):
        """Test full training pipeline with CatBoost on real multi-label data."""
        data = real_multi_label_classification_data
        output_dir = os.path.join(data["temp_dir"], "output_cb")

        vtune = VespaTune(
            train_filename=data["train_path"],
            valid_filename=data["valid_path"],
            output=output_dir,
            targets=data["targets"],
            task=data["task"],
            model_type="catboost",
            num_trials=2,
            time_limit=60,
        )

        vtune.train()

        model = joblib.load(os.path.join(output_dir, "vtune_model.final"))
        assert isinstance(model, list)
        assert len(model) == len(data["targets"])
