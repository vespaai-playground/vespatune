import json
import os

import joblib
import numpy as np
import pytest
import xgboost as xgb

from vespatune import VespaTuneExport, export_model
from vespatune.enums import ProblemType
from vespatune.models.catboost_model import CatBoostModel
from vespatune.models.lightgbm_model import LightGBMModel
from vespatune.models.xgboost_model import XGBoostModel
from vespatune.schemas import ModelConfig


@pytest.fixture
def trained_binary_model_for_export(binary_classification_data):
    """Create a trained binary classification model for export testing.

    This creates a simple gbtree model directly to ensure ONNX export works,
    since gblinear models are not supported by onnxmltools.
    """
    from vespatune import XGBoostPreprocessor

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

    # Create and save preprocessor
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

    return {"model_path": output_dir, "temp_dir": data["temp_dir"]}


@pytest.fixture
def trained_regression_model_for_export(regression_data):
    """Create a trained regression model for export testing."""
    from vespatune import XGBoostPreprocessor

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

    # Create and save preprocessor
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

    return {"model_path": output_dir, "temp_dir": data["temp_dir"]}


@pytest.fixture
def trained_multi_target_model_for_export(multi_target_regression_data):
    """Create a trained multi-target model for export testing."""
    from vespatune import XGBoostPreprocessor

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

    # Create and save preprocessor
    preprocessor = XGBoostPreprocessor(features=features, categorical_features=[])
    preprocessor.fit(
        train_df,
        problem_type=ProblemType.multi_column_regression,
        targets=data["targets"],
        idx="id",
    )

    joblib.dump(models, os.path.join(output_dir, "vtune_model.final"))
    joblib.dump(model_config, os.path.join(output_dir, "vtune.config"))
    preprocessor.save(os.path.join(output_dir, "vtune.preprocessor"))

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

        assert exporter.raw_model is not None
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


# ============================================================================
# ONNX vs Native Prediction Comparison Tests
# ============================================================================


class TestONNXPredictionAccuracyBinaryClassification:
    """Test that ONNX predictions match native model predictions for binary classification."""

    @pytest.fixture
    def real_data(self, real_binary_classification_data):
        """Prepare real data for model training."""
        data = real_binary_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]

        # Get numeric features only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["id", "income"]]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["income"].values)
        y_valid = le.transform(valid_df["income"].values)

        return X_train, y_train, X_valid, y_valid, len(feature_cols)

    def test_xgboost_onnx_predictions_match(self, real_data):
        """Test XGBoost ONNX predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = XGBoostModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "booster": "gbtree"}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # XGBoost ONNX output: [labels, probabilities]
        onnx_proba = onnx_output[1]

        # Convert list of dicts to array if needed
        if isinstance(onnx_proba, list) and isinstance(onnx_proba[0], dict):
            onnx_proba = np.array([[p[0], p[1]] for p in onnx_proba])

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)

    def test_lightgbm_onnx_predictions_match(self, real_data):
        """Test LightGBM ONNX predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = LightGBMModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # LightGBM ONNX output: [labels, probabilities]
        onnx_proba = onnx_output[1]

        # Convert list of dicts to array if needed
        if isinstance(onnx_proba, list) and isinstance(onnx_proba[0], dict):
            onnx_proba = np.array([[p[0], p[1]] for p in onnx_proba])

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)

    def test_catboost_onnx_predictions_match(self, real_data):
        """Test CatBoost ONNX predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = CatBoostModel(problem_type="binary_classification", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # CatBoost ONNX outputs: [labels, probabilities]
        # Probabilities are a list of dicts: [{0: prob0, 1: prob1}, ...]
        onnx_proba_raw = onnx_output[1]

        # Convert list of dicts to array
        if isinstance(onnx_proba_raw, list) and isinstance(onnx_proba_raw[0], dict):
            onnx_proba = np.array([[p[0], p[1]] for p in onnx_proba_raw])
        else:
            onnx_proba = np.array(onnx_proba_raw)

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)


class TestONNXPredictionAccuracyRegression:
    """Test that ONNX predictions match native model predictions for regression."""

    @pytest.fixture
    def real_data(self, real_single_column_regression_data):
        """Prepare real data for model training."""
        data = real_single_column_regression_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]

        # Get numeric features only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["id", "target"]]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)
        y_train = train_df["target"].values.astype(np.float32)
        y_valid = valid_df["target"].values.astype(np.float32)

        return X_train, y_train, X_valid, y_valid, len(feature_cols)

    def test_xgboost_onnx_predictions_match(self, real_data):
        """Test XGBoost ONNX regression predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = XGBoostModel(problem_type="single_column_regression", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "booster": "gbtree"}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_pred = model.predict(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        onnx_pred = onnx_output[0].ravel()

        assert np.allclose(native_pred, onnx_pred, atol=1e-5)

    def test_lightgbm_onnx_predictions_match(self, real_data):
        """Test LightGBM ONNX regression predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = LightGBMModel(problem_type="single_column_regression", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_pred = model.predict(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        onnx_pred = onnx_output[0].ravel()

        assert np.allclose(native_pred, onnx_pred, atol=1e-5)

    def test_catboost_onnx_predictions_match(self, real_data):
        """Test CatBoost ONNX regression predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = CatBoostModel(problem_type="single_column_regression", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_pred = model.predict(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        onnx_pred = onnx_output[0].ravel()

        assert np.allclose(native_pred, onnx_pred, atol=1e-5)


class TestONNXPredictionAccuracyMultiClass:
    """Test that ONNX predictions match native model predictions for multi-class classification."""

    @pytest.fixture
    def real_data(self, real_multi_class_classification_data):
        """Prepare real data for model training."""
        data = real_multi_class_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]

        feature_cols = [c for c in train_df.columns if c not in ["id", "target"]]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["target"].values)
        y_valid = le.transform(valid_df["target"].values)
        n_classes = len(le.classes_)

        return X_train, y_train, X_valid, y_valid, len(feature_cols), n_classes

    def test_xgboost_onnx_predictions_match(self, real_data):
        """Test XGBoost ONNX multi-class predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = XGBoostModel(problem_type="multi_class_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "booster": "gbtree"}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # XGBoost ONNX output: [labels, probabilities]
        onnx_proba = onnx_output[1]

        # Convert list of dicts to array if needed
        if isinstance(onnx_proba, list) and isinstance(onnx_proba[0], dict):
            onnx_proba = np.array([[p[i] for i in range(n_classes)] for p in onnx_proba])

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)

    def test_lightgbm_onnx_predictions_match(self, real_data):
        """Test LightGBM ONNX multi-class predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = LightGBMModel(problem_type="multi_class_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "num_class": n_classes}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # LightGBM ONNX output: [labels, probabilities]
        onnx_proba = onnx_output[1]

        # Convert list of dicts to array if needed
        if isinstance(onnx_proba, list) and isinstance(onnx_proba[0], dict):
            onnx_proba = np.array([[p[i] for i in range(n_classes)] for p in onnx_proba])

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)

    def test_catboost_onnx_predictions_match(self, real_data):
        """Test CatBoost ONNX multi-class predictions match native predictions."""
        import onnxruntime as ort

        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = CatBoostModel(problem_type="multi_class_classification", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)

        # Native predictions
        native_proba = model.predict_proba(X_valid)

        # ONNX export and predictions
        onnx_model = model.to_onnx(n_features=n_features)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = sess.get_inputs()[0].name
        onnx_output = sess.run(None, {input_name: X_valid})

        # CatBoost ONNX outputs: [labels, probabilities]
        # Probabilities are a list of dicts: [{0: prob0, 1: prob1, ...}, ...]
        onnx_proba_raw = onnx_output[1]

        # Convert list of dicts to array
        if isinstance(onnx_proba_raw, list) and isinstance(onnx_proba_raw[0], dict):
            onnx_proba = np.array([[p[i] for i in range(n_classes)] for p in onnx_proba_raw])
        else:
            onnx_proba = np.array(onnx_proba_raw)

        assert np.allclose(native_proba, onnx_proba, atol=1e-5)


class TestONNXPredictionAccuracyMultiLabel:
    """Test ONNX prediction accuracy for multi-label classification models.

    Note: Multi-label classification requires MultiOutputClassifier wrappers internally,
    and ONNX export for these wrapped models is not reliably supported. These tests are
    skipped as the ONNX converters don't properly handle multi-output model structures.
    """

    @pytest.fixture
    def real_data(self, real_multi_label_classification_data):
        """Prepare real multi-label classification data for ONNX tests."""
        data = real_multi_label_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]
        features = data["features"]
        targets = data["targets"]

        # Filter to numeric features only
        numeric_features = [f for f in features if train_df[f].dtype in ["int64", "float64"]]

        X_train = train_df[numeric_features].values.astype(np.float32)
        y_train = train_df[targets].values
        X_valid = valid_df[numeric_features].values.astype(np.float32)
        y_valid = valid_df[targets].values
        n_features = X_train.shape[1]
        n_labels = len(targets)
        return X_train, y_train, X_valid, y_valid, n_features, n_labels

    @pytest.mark.skip(reason="ONNX export for multi-label with MultiOutputClassifier not supported")
    def test_xgboost_onnx_predictions_match(self, real_data):
        """Test XGBoost ONNX multi-label predictions match native predictions."""
        pass

    @pytest.mark.skip(reason="ONNX export for multi-label with MultiOutputClassifier not supported")
    def test_lightgbm_onnx_predictions_match(self, real_data):
        """Test LightGBM ONNX multi-label predictions match native predictions."""
        pass

    @pytest.mark.skip(reason="ONNX export for multi-label with MultiOutputClassifier not supported")
    def test_catboost_onnx_predictions_match(self, real_data):
        """Test CatBoost ONNX multi-label predictions match native predictions."""
        pass


class TestONNXPredictionAccuracyMultiColumnRegression:
    """Test ONNX prediction accuracy for multi-column regression models.

    Note: Multi-column regression requires MultiOutputRegressor wrappers internally,
    and ONNX export for these wrapped models is not reliably supported. These tests are
    skipped as the ONNX converters don't properly handle multi-output model structures.
    """

    @pytest.fixture
    def real_data(self, real_multi_column_regression_data):
        """Prepare real multi-column regression data for ONNX tests."""
        data = real_multi_column_regression_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]
        features = data["features"]
        targets = data["targets"]

        # Filter to numeric features only
        numeric_features = [f for f in features if train_df[f].dtype in ["int64", "float64"]]

        X_train = train_df[numeric_features].values.astype(np.float32)
        y_train = train_df[targets].values.astype(np.float32)
        X_valid = valid_df[numeric_features].values.astype(np.float32)
        y_valid = valid_df[targets].values.astype(np.float32)
        n_features = X_train.shape[1]
        n_targets = len(targets)
        return X_train, y_train, X_valid, y_valid, n_features, n_targets

    @pytest.mark.skip(reason="ONNX export for multi-output regression not supported")
    def test_xgboost_onnx_predictions_match(self, real_data):
        """Test XGBoost ONNX multi-column regression predictions match native predictions."""
        pass

    @pytest.mark.skip(reason="ONNX export for multi-output regression not supported")
    def test_lightgbm_onnx_predictions_match(self, real_data):
        """Test LightGBM ONNX multi-column regression predictions match native predictions."""
        pass

    @pytest.mark.skip(reason="ONNX export for multi-output regression not supported")
    def test_catboost_onnx_predictions_match(self, real_data):
        """Test CatBoost ONNX multi-column regression predictions match native predictions."""
        pass
