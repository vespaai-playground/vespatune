import json
import os

import joblib
import pytest
import xgboost as xgb

from vespatune import VespaTuneExport, export_model
from vespatune.enums import ProblemType
from vespatune.schemas import ModelConfig


@pytest.fixture
def trained_binary_model_for_export(binary_classification_data):
    """Create a trained binary classification model for export testing.

    This creates a simple gbtree model directly to ensure ONNX export works,
    since gblinear models are not supported by onnxmltools.
    """
    data = binary_classification_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd

    train_df = pd.read_csv(data["train_path"])
    pd.read_csv(data["valid_path"])  # Load to verify file exists

    features = ["feature1", "feature2", "feature3"]
    X_train = train_df[features].values
    y_train = train_df["target"].values

    # Train a simple gbtree model
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        booster="gbtree",
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Create model config
    model_config = ModelConfig(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=None,
        idx="id",
        targets=["target"],
        problem_type=ProblemType.binary_classification,
        output=output_dir,
        features=features,
        use_gpu=False,
        seed=42,
        categorical_features=[],
        num_trials=2,
        time_limit=60,
    )

    # Save model and config
    joblib.dump(model, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))

    return {"model_path": output_dir, "temp_dir": data["temp_dir"]}


@pytest.fixture
def trained_regression_model_for_export(regression_data):
    """Create a trained regression model for export testing."""
    data = regression_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd

    train_df = pd.read_csv(data["train_path"])

    features = ["feature1", "feature2", "feature3"]
    X_train = train_df[features].values
    y_train = train_df["target"].values

    model = xgb.XGBRegressor(
        n_estimators=10,
        max_depth=3,
        booster="gbtree",
    )
    model.fit(X_train, y_train)

    model_config = ModelConfig(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=None,
        idx="id",
        targets=["target"],
        problem_type=ProblemType.single_column_regression,
        output=output_dir,
        features=features,
        use_gpu=False,
        seed=42,
        categorical_features=[],
        num_trials=2,
        time_limit=60,
    )

    joblib.dump(model, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))

    return {"model_path": output_dir, "temp_dir": data["temp_dir"]}


@pytest.fixture
def trained_multi_target_model_for_export(multi_target_regression_data):
    """Create a trained multi-target model for export testing."""
    data = multi_target_regression_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")
    os.makedirs(output_dir, exist_ok=True)

    import pandas as pd

    train_df = pd.read_csv(data["train_path"])

    features = ["feature1", "feature2", "feature3"]
    X_train = train_df[features].values

    # Train one model per target
    models = []
    for target in data["targets"]:
        y_train = train_df[target].values
        model = xgb.XGBRegressor(
            n_estimators=10,
            max_depth=3,
            booster="gbtree",
        )
        model.fit(X_train, y_train)
        models.append(model)

    model_config = ModelConfig(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=None,
        idx="id",
        targets=data["targets"],
        problem_type=ProblemType.multi_column_regression,
        output=output_dir,
        features=features,
        use_gpu=False,
        seed=42,
        categorical_features=[],
        num_trials=2,
        time_limit=60,
    )

    joblib.dump(models, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))

    return {
        "model_path": output_dir,
        "temp_dir": data["temp_dir"],
        "targets": data["targets"],
    }


class TestVespaTuneExportInit:
    @pytest.mark.slow
    def test_init_loads_model(self, trained_binary_model_for_export):
        """Test that VespaTuneExport loads the model correctly."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        assert exporter.model is not None
        assert exporter.model_config is not None


class TestONNXExport:
    @pytest.mark.slow
    def test_export_binary_model(self, trained_binary_model_for_export):
        """Test ONNX export for binary classification model."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        output_dir = os.path.join(trained_binary_model_for_export["temp_dir"], "onnx_output")
        exported_files = exporter.export_to_onnx(output_dir=output_dir)

        assert len(exported_files) == 1
        assert os.path.exists(exported_files[0])
        assert exported_files[0].endswith(".onnx")

    @pytest.mark.slow
    def test_export_regression_model(self, trained_regression_model_for_export):
        """Test ONNX export for regression model."""
        exporter = VespaTuneExport(model_path=trained_regression_model_for_export["model_path"])

        output_dir = os.path.join(trained_regression_model_for_export["temp_dir"], "onnx_output")
        exported_files = exporter.export_to_onnx(output_dir=output_dir)

        assert len(exported_files) == 1
        assert os.path.exists(exported_files[0])

    @pytest.mark.slow
    def test_export_multi_target_model(self, trained_multi_target_model_for_export):
        """Test ONNX export for multi-target regression model."""
        exporter = VespaTuneExport(model_path=trained_multi_target_model_for_export["model_path"])

        output_dir = os.path.join(trained_multi_target_model_for_export["temp_dir"], "onnx_output")
        exported_files = exporter.export_to_onnx(output_dir=output_dir)

        # Should export one model per target
        assert len(exported_files) == len(trained_multi_target_model_for_export["targets"])
        for filepath in exported_files:
            assert os.path.exists(filepath)

    @pytest.mark.slow
    def test_export_default_output_dir(self, trained_binary_model_for_export):
        """Test ONNX export with default output directory."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        exported_files = exporter.export_to_onnx()  # No output_dir specified

        expected_dir = os.path.join(trained_binary_model_for_export["model_path"], "onnx")
        assert os.path.exists(expected_dir)
        assert len(exported_files) == 1


class TestONNXMetadata:
    @pytest.mark.slow
    def test_metadata_exported(self, trained_binary_model_for_export):
        """Test that metadata is exported with ONNX model."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        output_dir = os.path.join(trained_binary_model_for_export["temp_dir"], "onnx_output")
        exporter.export_to_onnx(output_dir=output_dir)

        metadata_path = os.path.join(output_dir, "metadata.json")
        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "features" in metadata
        assert "feature_mapping" in metadata
        assert "targets" in metadata
        assert "problem_type" in metadata
        assert "categorical_features" in metadata

    @pytest.mark.slow
    def test_feature_mapping(self, trained_binary_model_for_export):
        """Test that feature mapping correctly maps ONNX names to original names."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        output_dir = os.path.join(trained_binary_model_for_export["temp_dir"], "onnx_output")
        exporter.export_to_onnx(output_dir=output_dir)

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        feature_mapping = metadata["feature_mapping"]
        features = metadata["features"]

        # Check mapping has correct number of entries
        assert len(feature_mapping) == len(features)

        # Check mapping follows f%d pattern and maps to correct features
        for i, feature_name in enumerate(features):
            onnx_name = f"f{i}"
            assert onnx_name in feature_mapping
            assert feature_mapping[onnx_name] == feature_name


class TestONNXVerification:
    @pytest.mark.slow
    def test_export_with_verification(self, trained_binary_model_for_export):
        """Test ONNX export with verification enabled."""
        exporter = VespaTuneExport(model_path=trained_binary_model_for_export["model_path"])

        output_dir = os.path.join(trained_binary_model_for_export["temp_dir"], "onnx_output")

        # This should not raise an error if onnxruntime is installed
        try:
            exported_files = exporter.export_to_onnx(output_dir=output_dir, verify=True)
            assert len(exported_files) == 1
        except ImportError:
            # onnxruntime not installed, verification will be skipped
            pytest.skip("onnxruntime not installed for verification")


class TestExportModelFunction:
    @pytest.mark.slow
    def test_export_model_convenience_function(self, trained_binary_model_for_export):
        """Test the export_model convenience function."""
        output_dir = os.path.join(trained_binary_model_for_export["temp_dir"], "onnx_output")

        exported_files = export_model(
            model_path=trained_binary_model_for_export["model_path"],
            output_dir=output_dir,
        )

        assert len(exported_files) == 1
        assert os.path.exists(exported_files[0])
