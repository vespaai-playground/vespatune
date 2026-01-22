import os

import joblib
import pandas as pd
import pytest
import xgboost as xgb

from vespatune import VespaTune, VespaTuneExport, VespaTuneONNXPredict, VespaTunePredict, VespaTuneProcessor
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
        assert predictor.preprocessor is not None

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

    # Create and save preprocessor
    from vespatune import XGBoostPreprocessor

    preprocessor = XGBoostPreprocessor(features=features, categorical_features=[])
    preprocessor.fit(
        train_df,
        problem_type=ProblemType.binary_classification,
        targets=["target"],
        idx="id",
    )

    # Save model, config, and preprocessor
    joblib.dump(model, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))
    preprocessor.save(os.path.join(output_dir, "vtune.preprocessor"))

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

    # Create and save preprocessor
    from vespatune import XGBoostPreprocessor

    preprocessor = XGBoostPreprocessor(features=features, categorical_features=[])
    preprocessor.fit(
        train_df,
        problem_type=ProblemType.single_column_regression,
        targets=["target"],
        idx="id",
    )

    joblib.dump(model, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))
    preprocessor.save(os.path.join(output_dir, "vtune.preprocessor"))

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


# VespaTuneProcessor Tests


class TestVespaTuneProcessorInit:
    def test_init_from_model_dir(self, onnx_binary_model):
        """Test loading processor from model directory (with vtune.config)."""
        # onnx_binary_model fixture creates both model dir and onnx dir
        # The model dir has vtune.config
        model_dir = os.path.dirname(onnx_binary_model["onnx_path"])
        model_dir = os.path.join(model_dir, "trained_model")

        processor = VespaTuneProcessor(model_path=model_dir)

        assert processor.features is not None
        assert processor.preprocessor is not None

    def test_init_from_onnx_dir(self, onnx_binary_model):
        """Test loading processor from ONNX export directory (with metadata.json)."""
        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        assert processor.features == onnx_binary_model["features"]
        assert processor.preprocessor is not None
        assert processor.problem_type == "binary_classification"

    def test_init_invalid_path(self, tmp_path):
        """Test that invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VespaTuneProcessor(model_path=str(tmp_path))


class TestVespaTuneProcessorTransform:
    def test_transform_dataframe(self, onnx_binary_model):
        """Test transforming a DataFrame."""
        import numpy as np

        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        test_df = pd.read_csv(onnx_binary_model["test_path"])
        processed = processor.transform(test_df)

        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert processed.shape[0] == len(test_df)

    def test_transform_single(self, onnx_binary_model):
        """Test transforming a single sample."""
        import numpy as np

        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        sample = {"feature1": 0.5, "feature2": -0.3, "feature3": 1.2}
        processed = processor.transform_single(sample)

        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.float32
        assert processed.shape[0] == 1


class TestVespaTuneProcessorMetadata:
    def test_get_feature_names(self, onnx_binary_model):
        """Test getting input feature names."""
        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        features = processor.get_feature_names()

        assert features == onnx_binary_model["features"]

    def test_get_categorical_features(self, onnx_binary_model):
        """Test getting categorical feature names."""
        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        cat_features = processor.get_categorical_features()

        assert isinstance(cat_features, list)

    def test_get_feature_names_out(self, onnx_binary_model):
        """Test getting output feature names after transformation."""
        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        features_out = processor.get_feature_names_out()

        assert isinstance(features_out, list)
        assert len(features_out) > 0

    def test_get_input_schema(self, onnx_binary_model):
        """Test getting Pydantic input schema."""
        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        schema = processor.get_input_schema()

        assert schema is not None
        assert hasattr(schema, "model_fields")
        for feat in onnx_binary_model["features"]:
            assert feat in schema.model_fields


class TestVespaTuneProcessorIntegration:
    def test_processor_output_matches_onnx_input(self, onnx_binary_model):
        """Test that processor output can be passed directly to ONNX runtime."""
        import onnxruntime as ort

        processor = VespaTuneProcessor(model_path=onnx_binary_model["onnx_path"])

        # Transform test data
        test_df = pd.read_csv(onnx_binary_model["test_path"])
        processed = processor.transform(test_df)

        # Load ONNX model and run inference
        onnx_path = os.path.join(onnx_binary_model["onnx_path"], "model.onnx")
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # This should not raise an error
        outputs = session.run(None, {input_name: processed})

        assert outputs is not None
        assert len(outputs) > 0
