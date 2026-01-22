from unittest.mock import MagicMock

import numpy as np
import pytest

from vespatune.models import MODEL_REGISTRY, get_model, list_models, register_model
from vespatune.models.base import BaseModel
from vespatune.models.catboost_model import CatBoostModel
from vespatune.models.lightgbm_model import LightGBMModel
from vespatune.models.xgboost_model import XGBoostModel


class TestModelRegistry:
    def test_list_models(self):
        models = list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "catboost" in models

    def test_get_model_xgboost(self):
        model = get_model("xgboost", "binary_classification")
        assert isinstance(model, XGBoostModel)
        assert model.name == "xgboost"

    def test_get_model_lightgbm(self):
        model = get_model("lightgbm", "binary_classification")
        assert isinstance(model, LightGBMModel)
        assert model.name == "lightgbm"

    def test_get_model_catboost(self):
        model = get_model("catboost", "binary_classification")
        assert isinstance(model, CatBoostModel)
        assert model.name == "catboost"

    def test_get_model_case_insensitive(self):
        model = get_model("XGBOOST", "binary_classification")
        assert isinstance(model, XGBoostModel)

    def test_get_model_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("unknown_model", "binary_classification")

    def test_register_model(self):
        class DummyModel(BaseModel):
            name = "dummy"
            supports_categorical = False
            supports_gpu = False

            def get_params(self, trial, model_config):
                return {}

            def fit(self, X_train, y_train, X_valid, y_valid, params, categorical_features=None):
                pass

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.zeros((len(X), 2))

            def get_model(self):
                return None

            def save(self, path):
                pass

            def load(self, path):
                pass

            def to_onnx(self, n_features):
                raise NotImplementedError("ONNX export not supported for DummyModel")

        register_model("dummy", DummyModel)
        assert "dummy" in MODEL_REGISTRY

        model = get_model("dummy", "binary_classification")
        assert isinstance(model, DummyModel)

        # Cleanup
        del MODEL_REGISTRY["dummy"]


class TestXGBoostModel:
    @pytest.fixture
    def xgb_model(self):
        return XGBoostModel(problem_type="binary_classification", random_state=42)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randint(0, 2, 20)
        return X_train, y_train, X_valid, y_valid

    def test_model_attributes(self, xgb_model):
        assert xgb_model.name == "xgboost"
        assert xgb_model.supports_categorical is False
        assert xgb_model.supports_gpu is True

    def test_fit_predict_binary(self, xgb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        xgb_model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = xgb_model.predict(X_valid)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_binary(self, xgb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        xgb_model.fit(X_train, y_train, X_valid, y_valid, params)
        proba = xgb_model.predict_proba(X_valid)

        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_get_model(self, xgb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        xgb_model.fit(X_train, y_train, X_valid, y_valid, params)
        model = xgb_model.get_model()

        assert model is not None

    def test_regression(self):
        model = XGBoostModel(problem_type="single_column_regression", random_state=42)
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randn(20)

        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (20,)


class TestLightGBMModel:
    @pytest.fixture
    def lgb_model(self):
        return LightGBMModel(problem_type="binary_classification", random_state=42)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randint(0, 2, 20)
        return X_train, y_train, X_valid, y_valid

    def test_model_attributes(self, lgb_model):
        assert lgb_model.name == "lightgbm"
        assert lgb_model.supports_categorical is True
        assert lgb_model.supports_gpu is True

    def test_fit_predict_binary(self, lgb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        lgb_model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = lgb_model.predict(X_valid)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_binary(self, lgb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        lgb_model.fit(X_train, y_train, X_valid, y_valid, params)
        proba = lgb_model.predict_proba(X_valid)

        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_regression(self):
        model = LightGBMModel(problem_type="single_column_regression", random_state=42)
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randn(20)

        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (20,)


class TestCatBoostModel:
    @pytest.fixture
    def cb_model(self):
        return CatBoostModel(problem_type="binary_classification", random_state=42)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randint(0, 2, 20)
        return X_train, y_train, X_valid, y_valid

    def test_model_attributes(self, cb_model):
        assert cb_model.name == "catboost"
        assert cb_model.supports_categorical is True
        assert cb_model.supports_gpu is True

    def test_fit_predict_binary(self, cb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        cb_model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = cb_model.predict(X_valid)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_binary(self, cb_model, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        cb_model.fit(X_train, y_train, X_valid, y_valid, params)
        proba = cb_model.predict_proba(X_valid)

        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_regression(self):
        model = CatBoostModel(problem_type="single_column_regression", random_state=42)
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randn(20)

        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}
        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (20,)


class TestToOnnx:
    """Test ONNX export for each model type."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_valid = np.random.randn(20, 5)
        y_valid = np.random.randint(0, 2, 20)
        return X_train, y_train, X_valid, y_valid

    def test_xgboost_to_onnx(self, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        model = XGBoostModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "booster": "gbtree"}
        model.fit(X_train, y_train, X_valid, y_valid, params)

        onnx_model = model.to_onnx(n_features=5)
        assert onnx_model is not None
        assert hasattr(onnx_model, "graph")

    def test_lightgbm_to_onnx(self, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        model = LightGBMModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}
        model.fit(X_train, y_train, X_valid, y_valid, params)

        onnx_model = model.to_onnx(n_features=5)
        assert onnx_model is not None
        assert hasattr(onnx_model, "graph")

    def test_catboost_to_onnx(self, sample_data):
        X_train, y_train, X_valid, y_valid = sample_data
        model = CatBoostModel(problem_type="binary_classification", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}
        model.fit(X_train, y_train, X_valid, y_valid, params)

        onnx_model = model.to_onnx(n_features=5)
        assert onnx_model is not None
        assert hasattr(onnx_model, "graph")


class TestGetParams:
    def test_xgboost_get_params(self):
        model = XGBoostModel(problem_type="binary_classification", random_state=42)
        trial = MagicMock()
        trial.suggest_float.return_value = 0.1
        trial.suggest_int.return_value = 5
        trial.suggest_categorical.return_value = "gbtree"

        model_config = MagicMock()
        model_config.use_gpu = False

        params = model.get_params(trial, model_config)
        assert "learning_rate" in params or trial.suggest_float.called

    def test_lightgbm_get_params(self):
        model = LightGBMModel(problem_type="binary_classification", random_state=42)
        trial = MagicMock()
        trial.suggest_float.return_value = 0.1
        trial.suggest_int.return_value = 5
        trial.suggest_categorical.return_value = "gbdt"

        model_config = MagicMock()
        model_config.use_gpu = False

        model.get_params(trial, model_config)
        assert trial.suggest_float.called or trial.suggest_int.called

    def test_catboost_get_params(self):
        model = CatBoostModel(problem_type="binary_classification", random_state=42)
        trial = MagicMock()
        trial.suggest_float.return_value = 0.1
        trial.suggest_int.return_value = 5
        trial.suggest_categorical.return_value = "SymmetricTree"

        model_config = MagicMock()
        model_config.use_gpu = False

        model.get_params(trial, model_config)
        assert trial.suggest_float.called or trial.suggest_int.called


# ============================================================================
# Real Data Tests - Binary Classification
# ============================================================================


class TestRealDataBinaryClassification:
    """Test all models with real binary classification data."""

    @pytest.fixture
    def real_data(self, real_binary_classification_data):
        """Prepare real data for model training."""
        data = real_binary_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]

        # Get numeric features only for simplicity
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["id", "income"]]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)

        # Encode target
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["income"].values)
        y_valid = le.transform(valid_df["income"].values)

        return X_train, y_train, X_valid, y_valid, len(feature_cols)

    def test_xgboost_binary_real_data(self, real_data):
        """Test XGBoost with real binary classification data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = XGBoostModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_lightgbm_binary_real_data(self, real_data):
        """Test LightGBM with real binary classification data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = LightGBMModel(problem_type="binary_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_catboost_binary_real_data(self, real_data):
        """Test CatBoost with real binary classification data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = CatBoostModel(problem_type="binary_classification", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ============================================================================
# Real Data Tests - Multi-Class Classification
# ============================================================================


class TestRealDataMultiClassClassification:
    """Test all models with real multi-class classification data (Iris)."""

    @pytest.fixture
    def real_data(self, real_multi_class_classification_data):
        """Prepare real data for model training."""
        data = real_multi_class_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]

        feature_cols = [c for c in train_df.columns if c not in ["id", "target"]]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)

        # Encode target
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_train = le.fit_transform(train_df["target"].values)
        y_valid = le.transform(valid_df["target"].values)
        n_classes = len(le.classes_)

        return X_train, y_train, X_valid, y_valid, len(feature_cols), n_classes

    def test_xgboost_multiclass_real_data(self, real_data):
        """Test XGBoost with real multi-class classification data."""
        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = XGBoostModel(problem_type="multi_class_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_lightgbm_multiclass_real_data(self, real_data):
        """Test LightGBM with real multi-class classification data."""
        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = LightGBMModel(problem_type="multi_class_classification", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "num_class": n_classes}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_catboost_multiclass_real_data(self, real_data):
        """Test CatBoost with real multi-class classification data."""
        X_train, y_train, X_valid, y_valid, n_features, n_classes = real_data
        model = CatBoostModel(problem_type="multi_class_classification", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)
        proba = model.predict_proba(X_valid)

        # CatBoost predict returns (n_samples, 1) for multiclass, flatten it
        predictions = predictions.ravel()
        assert predictions.shape == (len(y_valid),)
        assert proba.shape == (len(y_valid), n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ============================================================================
# Real Data Tests - Single Column Regression
# ============================================================================


class TestRealDataSingleColumnRegression:
    """Test all models with real single column regression data."""

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

    def test_xgboost_regression_real_data(self, real_data):
        """Test XGBoost with real regression data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = XGBoostModel(problem_type="single_column_regression", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (len(y_valid),)

    def test_lightgbm_regression_real_data(self, real_data):
        """Test LightGBM with real regression data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = LightGBMModel(problem_type="single_column_regression", random_state=42)
        params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (len(y_valid),)

    def test_catboost_regression_real_data(self, real_data):
        """Test CatBoost with real regression data."""
        X_train, y_train, X_valid, y_valid, n_features = real_data
        model = CatBoostModel(problem_type="single_column_regression", random_state=42)
        params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

        model.fit(X_train, y_train, X_valid, y_valid, params)
        predictions = model.predict(X_valid)

        assert predictions.shape == (len(y_valid),)


# ============================================================================
# Real Data Tests - Multi-Column Regression
# ============================================================================


class TestRealDataMultiColumnRegression:
    """Test all models with real multi-column regression data."""

    @pytest.fixture
    def real_data(self, real_multi_column_regression_data):
        """Prepare real data for model training."""
        data = real_multi_column_regression_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]
        target_cols = data["targets"]

        # Get numeric features only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["id"] + target_cols]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_cols].values.astype(np.float32)
        y_valid = valid_df[target_cols].values.astype(np.float32)

        return X_train, y_train, X_valid, y_valid, len(feature_cols), len(target_cols)

    def test_xgboost_multi_regression_real_data(self, real_data):
        """Test XGBoost with real multi-column regression data (one model per target)."""
        X_train, y_train, X_valid, y_valid, n_features, n_targets = real_data

        for idx in range(n_targets):
            model = XGBoostModel(problem_type="multi_column_regression", random_state=42)
            params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)

            assert predictions.shape == (len(y_valid),)

    def test_lightgbm_multi_regression_real_data(self, real_data):
        """Test LightGBM with real multi-column regression data (one model per target)."""
        X_train, y_train, X_valid, y_valid, n_features, n_targets = real_data

        for idx in range(n_targets):
            model = LightGBMModel(problem_type="multi_column_regression", random_state=42)
            params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)

            assert predictions.shape == (len(y_valid),)

    def test_catboost_multi_regression_real_data(self, real_data):
        """Test CatBoost with real multi-column regression data (one model per target)."""
        X_train, y_train, X_valid, y_valid, n_features, n_targets = real_data

        for idx in range(n_targets):
            model = CatBoostModel(problem_type="multi_column_regression", random_state=42)
            params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)

            assert predictions.shape == (len(y_valid),)


# ============================================================================
# Real Data Tests - Multi-Label Classification
# ============================================================================


class TestRealDataMultiLabelClassification:
    """Test all models with real multi-label classification data."""

    @pytest.fixture
    def real_data(self, real_multi_label_classification_data):
        """Prepare real data for model training."""
        data = real_multi_label_classification_data
        train_df = data["train_df"]
        valid_df = data["valid_df"]
        target_cols = data["targets"]

        # Get numeric features only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["id"] + target_cols]

        X_train = train_df[feature_cols].values.astype(np.float32)
        X_valid = valid_df[feature_cols].values.astype(np.float32)
        y_train = train_df[target_cols].values.astype(np.int32)
        y_valid = valid_df[target_cols].values.astype(np.int32)

        return X_train, y_train, X_valid, y_valid, len(feature_cols), len(target_cols)

    def test_xgboost_multilabel_real_data(self, real_data):
        """Test XGBoost with real multi-label classification data (one model per label)."""
        X_train, y_train, X_valid, y_valid, n_features, n_labels = real_data

        for idx in range(n_labels):
            model = XGBoostModel(problem_type="multi_label_classification", random_state=42)
            params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)
            proba = model.predict_proba(X_valid)

            assert predictions.shape == (len(y_valid),)
            assert proba.shape == (len(y_valid), 2)

    def test_lightgbm_multilabel_real_data(self, real_data):
        """Test LightGBM with real multi-label classification data (one model per label)."""
        X_train, y_train, X_valid, y_valid, n_features, n_labels = real_data

        for idx in range(n_labels):
            model = LightGBMModel(problem_type="multi_label_classification", random_state=42)
            params = {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)
            proba = model.predict_proba(X_valid)

            assert predictions.shape == (len(y_valid),)
            assert proba.shape == (len(y_valid), 2)

    def test_catboost_multilabel_real_data(self, real_data):
        """Test CatBoost with real multi-label classification data (one model per label)."""
        X_train, y_train, X_valid, y_valid, n_features, n_labels = real_data

        for idx in range(n_labels):
            model = CatBoostModel(problem_type="multi_label_classification", random_state=42)
            params = {"iterations": 10, "depth": 3, "learning_rate": 0.1}

            model.fit(X_train, y_train[:, idx], X_valid, y_valid[:, idx], params)
            predictions = model.predict(X_valid)
            proba = model.predict_proba(X_valid)

            assert predictions.shape == (len(y_valid),)
            assert proba.shape == (len(y_valid), 2)
