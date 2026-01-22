from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnx
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""

    name = "lightgbm"
    supports_categorical = True  # Native categorical support
    supports_gpu = True

    def __init__(self, problem_type: str, random_state: int = 42):
        super().__init__(problem_type, random_state)
        self.model = None

    def get_params(self, trial, model_config) -> Dict[str, Any]:
        """Get LightGBM hyperparameters for Optuna trial."""
        # Boosting type must be selected first as it affects other params
        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "random_state": self.random_state,
            "verbosity": -1,
            "boosting_type": boosting_type,
        }

        # Early stopping not available in dart mode
        if boosting_type != "dart":
            params["early_stopping_rounds"] = trial.suggest_int("early_stopping_rounds", 50, 300)

        # GOSS-specific parameters (GOSS cannot use bagging/subsample)
        if boosting_type == "goss":
            params["top_rate"] = trial.suggest_float("top_rate", 0.1, 0.5)
            params["other_rate"] = trial.suggest_float("other_rate", 0.01, 0.2)
        else:
            # Bagging parameters only for gbdt and dart
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["subsample_freq"] = trial.suggest_int("subsample_freq", 1, 10)

        # DART-specific parameters
        if boosting_type == "dart":
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.0, 0.3)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)
            params["max_drop"] = trial.suggest_int("max_drop", 10, 100)

        # GPU settings
        if model_config.use_gpu:
            params["device"] = "gpu"

        # Extra tree parameters (only for gbdt)
        if boosting_type == "gbdt":
            params["extra_trees"] = trial.suggest_categorical("extra_trees", [True, False])

        params["path_smooth"] = trial.suggest_float("path_smooth", 0.0, 1.0)

        # Feature fraction parameters
        params["feature_fraction_bynode"] = trial.suggest_float("feature_fraction_bynode", 0.3, 1.0)

        # Task-specific parameters
        params = self._add_task_specific_params(trial, params)

        return params

    def _add_task_specific_params(self, trial, params: Dict) -> Dict:
        """Add task-specific parameters."""
        if self.problem_type == "binary_classification":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            params["is_unbalance"] = trial.suggest_categorical("is_unbalance", [True, False])
            if not params["is_unbalance"]:
                params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 10.0, log=True)

        elif self.problem_type == "multi_class_classification":
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"

        elif self.problem_type == "multi_label_classification":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            objective = trial.suggest_categorical(
                "objective", ["regression", "regression_l1", "huber", "fair", "quantile"]
            )
            params["objective"] = objective

            if objective == "huber":
                params["alpha"] = trial.suggest_float("huber_alpha", 0.5, 2.0)
            elif objective == "fair":
                params["fair_c"] = trial.suggest_float("fair_c", 0.1, 2.0)
            elif objective == "quantile":
                params["alpha"] = trial.suggest_float("quantile_alpha", 0.1, 0.9)

            params["metric"] = "rmse"

        return params

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        params: Dict[str, Any],
        categorical_features: Optional[List[int]] = None,
    ) -> None:
        """Train LightGBM model."""
        import lightgbm as lgb

        # Extract early stopping rounds (not available in dart mode)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)

        # Build callbacks - early stopping only if not dart mode
        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )

        # Create model based on problem type
        if self.problem_type in ("binary_classification", "multi_class_classification"):
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        # Prepare categorical features
        cat_feature = categorical_features if categorical_features else "auto"

        # Fit model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=callbacks,
            categorical_feature=cat_feature,
        )

        self.best_iteration = getattr(self.model, "best_iteration_", self.model.n_estimators)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        """Get the underlying LightGBM model."""
        return self.model

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

    def get_booster(self):
        """Get the LightGBM booster for ONNX export."""
        return self.model.booster_

    def to_onnx(self, n_features: int) -> onnx.ModelProto:
        """Convert LightGBM model to ONNX format."""
        initial_types = [("input", FloatTensorType([None, n_features]))]
        return convert_lightgbm(self.model, initial_types=initial_types, target_opset=15)
