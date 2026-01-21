import os

import joblib
import pandas as pd
import pytest
import xgboost as xgb

from vespatune import VespaTune, VespaTuneExport, VespaTuneONNXPredict, VespaTunePredict
from vespatune.enums import ProblemType
from vespatune.schemas import ModelConfig


@pytest.fixture
def trained_binary_model(binary_classification_data):
    """Create a trained binary classification model for testing."""
    data = binary_classification_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")

    vtune = VespaTune(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=data["test_path"],
        output=output_dir,
        targets=data["targets"],
        task=data["task"],
        num_trials=2,
        time_limit=60,
    )

    vtune.train()

    return {
        "model_path": output_dir,
        "test_path": data["test_path"],
        "temp_dir": data["temp_dir"],
    }


@pytest.fixture
def trained_regression_model(regression_data):
    """Create a trained regression model for testing."""
    data = regression_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")

    vtune = VespaTune(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=data["test_path"],
        output=output_dir,
        targets=data["targets"],
        task=data["task"],
        num_trials=2,
        time_limit=60,
    )

    vtune.train()

    return {
        "model_path": output_dir,
        "test_path": data["test_path"],
        "temp_dir": data["temp_dir"],
    }


@pytest.fixture
def trained_multi_target_model(multi_target_regression_data):
    """Create a trained multi-target regression model for testing."""
    data = multi_target_regression_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")

    vtune = VespaTune(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=data["test_path"],
        output=output_dir,
        targets=data["targets"],
        task=data["task"],
        num_trials=2,
        time_limit=60,
    )

    vtune.train()

    return {
        "model_path": output_dir,
        "test_path": data["test_path"],
        "temp_dir": data["temp_dir"],
        "targets": data["targets"],
    }


class TestVespaTunePredictInit:
    @pytest.mark.slow
    def test_init_loads_model(self, trained_binary_model):
        """Test that VespaTunePredict loads the model correctly."""
        predictor = VespaTunePredict(model_path=trained_binary_model["model_path"])

        assert predictor.model is not None
        assert predictor.model_config is not None
        assert predictor.categorical_encoder is not None

    @pytest.mark.slow
    def test_init_loads_config(self, trained_binary_model):
        """Test that VespaTunePredict loads configuration correctly."""
        predictor = VespaTunePredict(model_path=trained_binary_model["model_path"])

        assert predictor.model_config.targets == ["target"]


class TestVespaTunePredictFile:
    @pytest.mark.slow
    def test_predict_file_binary(self, trained_binary_model):
        """Test file-based prediction for binary classification."""
        predictor = VespaTunePredict(model_path=trained_binary_model["model_path"])

        output_path = os.path.join(trained_binary_model["temp_dir"], "predictions.csv")
        predictor.predict_file(
            test_filename=trained_binary_model["test_path"],
            out_filename=output_path,
        )

        assert os.path.exists(output_path)
        predictions = pd.read_csv(output_path)
        assert len(predictions) > 0
        assert "id" in predictions.columns

    @pytest.mark.slow
    def test_predict_file_regression(self, trained_regression_model):
        """Test file-based prediction for regression."""
        predictor = VespaTunePredict(model_path=trained_regression_model["model_path"])

        output_path = os.path.join(trained_regression_model["temp_dir"], "predictions.csv")
        predictor.predict_file(
            test_filename=trained_regression_model["test_path"],
            out_filename=output_path,
        )

        assert os.path.exists(output_path)
        predictions = pd.read_csv(output_path)
        assert "target" in predictions.columns

    @pytest.mark.slow
    def test_predict_file_multi_target(self, trained_multi_target_model):
        """Test file-based prediction for multi-target regression."""
        predictor = VespaTunePredict(model_path=trained_multi_target_model["model_path"])

        output_path = os.path.join(trained_multi_target_model["temp_dir"], "predictions.csv")
        predictor.predict_file(
            test_filename=trained_multi_target_model["test_path"],
            out_filename=output_path,
        )

        assert os.path.exists(output_path)
        predictions = pd.read_csv(output_path)
        for target in trained_multi_target_model["targets"]:
            assert target in predictions.columns


class TestVespaTunePredictSingle:
    @pytest.mark.slow
    def test_predict_single_binary(self, trained_binary_model):
        """Test single sample prediction for binary classification."""
        predictor = VespaTunePredict(model_path=trained_binary_model["model_path"])

        sample = {
            "feature1": 0.5,
            "feature2": -0.3,
            "feature3": 1.2,
            "cat_feature": "A",
        }

        prediction = predictor.predict_single(sample)

        assert prediction is not None
        assert isinstance(prediction, dict)

    @pytest.mark.slow
    def test_predict_single_regression(self, trained_regression_model):
        """Test single sample prediction for regression."""
        predictor = VespaTunePredict(model_path=trained_regression_model["model_path"])

        sample = {
            "feature1": 0.5,
            "feature2": -0.3,
            "feature3": 1.2,
        }

        prediction = predictor.predict_single(sample)

        assert prediction is not None
        assert "target" in prediction


class TestPredictionSchema:
    @pytest.mark.slow
    def test_get_prediction_schema(self, trained_binary_model):
        """Test getting prediction schema for API."""
        predictor = VespaTunePredict(model_path=trained_binary_model["model_path"])

        schema = predictor.get_prediction_schema()

        assert schema is not None
        # Schema should be a Pydantic model class
        assert hasattr(schema, "model_fields")


# ONNX Inference Tests


@pytest.fixture
def onnx_binary_model(binary_classification_data):
    """Create a trained binary classification model and export to ONNX."""
    data = binary_classification_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")
    onnx_dir = os.path.join(data["temp_dir"], "onnx_model")
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(data["train_path"])

    features = ["feature1", "feature2", "feature3"]
    X_train = train_df[features].values
    y_train = train_df["target"].values

    # Train a simple gbtree model
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        booster="gbtree",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Create model config
    model_config = ModelConfig(
        train_filename=data["train_path"],
        valid_filename=data["valid_path"],
        test_filename=data["test_path"],
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
    joblib.dump(None, os.path.join(output_dir, "vtune.categorical_encoder"))
    joblib.dump(None, os.path.join(output_dir, "vtune.target_encoder"))

    # Export to ONNX
    exporter = VespaTuneExport(model_path=output_dir)
    exporter.export_to_onnx(output_dir=onnx_dir)

    return {
        "onnx_path": onnx_dir,
        "test_path": data["test_path"],
        "temp_dir": data["temp_dir"],
        "features": features,
    }


@pytest.fixture
def onnx_regression_model(regression_data):
    """Create a trained regression model and export to ONNX."""
    data = regression_data
    output_dir = os.path.join(data["temp_dir"], "trained_model")
    onnx_dir = os.path.join(data["temp_dir"], "onnx_model")
    os.makedirs(output_dir, exist_ok=True)

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
        test_filename=data["test_path"],
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
    joblib.dump(None, os.path.join(output_dir, "vtune.categorical_encoder"))
    joblib.dump(None, os.path.join(output_dir, "vtune.target_encoder"))

    exporter = VespaTuneExport(model_path=output_dir)
    exporter.export_to_onnx(output_dir=onnx_dir)

    return {
        "onnx_path": onnx_dir,
        "test_path": data["test_path"],
        "temp_dir": data["temp_dir"],
        "features": features,
    }


class TestVespaTuneONNXPredictInit:
    def test_init_loads_onnx_model(self, onnx_binary_model):
        """Test that VespaTuneONNXPredict loads the ONNX model correctly."""
        predictor = VespaTuneONNXPredict(model_path=onnx_binary_model["onnx_path"])

        assert predictor.sessions is not None
        assert len(predictor.sessions) == 1
        assert predictor.features == onnx_binary_model["features"]

    def test_init_loads_metadata(self, onnx_binary_model):
        """Test that VespaTuneONNXPredict loads metadata correctly."""
        predictor = VespaTuneONNXPredict(model_path=onnx_binary_model["onnx_path"])

        assert predictor.targets == ["target"]
        assert predictor.problem_type == "binary_classification"
        assert predictor.feature_mapping is not None


class TestVespaTuneONNXPredictSingle:
    def test_predict_single_binary(self, onnx_binary_model):
        """Test single sample ONNX prediction for binary classification."""
        predictor = VespaTuneONNXPredict(model_path=onnx_binary_model["onnx_path"])

        sample = {
            "feature1": 0.5,
            "feature2": -0.3,
            "feature3": 1.2,
        }

        prediction = predictor.predict_single(sample)

        assert prediction is not None
        assert isinstance(prediction, dict)
        assert "id" in prediction

    def test_predict_single_regression(self, onnx_regression_model):
        """Test single sample ONNX prediction for regression."""
        predictor = VespaTuneONNXPredict(model_path=onnx_regression_model["onnx_path"])

        sample = {
            "feature1": 0.5,
            "feature2": -0.3,
            "feature3": 1.2,
        }

        prediction = predictor.predict_single(sample)

        assert prediction is not None
        assert "target" in prediction


class TestVespaTuneONNXPredictFile:
    def test_predict_file_binary(self, onnx_binary_model):
        """Test file-based ONNX prediction for binary classification."""
        predictor = VespaTuneONNXPredict(model_path=onnx_binary_model["onnx_path"])

        output_path = os.path.join(onnx_binary_model["temp_dir"], "onnx_predictions.csv")
        predictor.predict_file(
            test_filename=onnx_binary_model["test_path"],
            out_filename=output_path,
        )

        assert os.path.exists(output_path)
        predictions = pd.read_csv(output_path)
        assert len(predictions) > 0
        assert "id" in predictions.columns

    def test_predict_file_regression(self, onnx_regression_model):
        """Test file-based ONNX prediction for regression."""
        predictor = VespaTuneONNXPredict(model_path=onnx_regression_model["onnx_path"])

        output_path = os.path.join(onnx_regression_model["temp_dir"], "onnx_predictions.csv")
        predictor.predict_file(
            test_filename=onnx_regression_model["test_path"],
            out_filename=output_path,
        )

        assert os.path.exists(output_path)
        predictions = pd.read_csv(output_path)
        assert "target" in predictions.columns


class TestONNXPredictionSchema:
    def test_get_prediction_schema(self, onnx_binary_model):
        """Test getting prediction schema from ONNX predictor."""
        predictor = VespaTuneONNXPredict(model_path=onnx_binary_model["onnx_path"])

        schema = predictor.get_prediction_schema()

        assert schema is not None
        assert hasattr(schema, "model_fields")
