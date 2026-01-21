from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import xgboost as xgb

from .base import BaseModel


xgb.set_config(verbosity=0)


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    name = "xgboost"
    supports_categorical = False  # Requires encoding
    supports_gpu = True

    def __init__(self, problem_type: str, random_state: int = 42):
        super().__init__(problem_type, random_state)
        self.model = None

    def get_params(self, trial, model_config) -> Dict[str, Any]:
        """Get XGBoost hyperparameters for Optuna trial."""
        # Base parameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "random_state": self.random_state,
        }

        # GPU settings
        if model_config.use_gpu:
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        else:
            params["tree_method"] = trial.suggest_categorical("tree_method", ["hist", "approx"])

        # Booster selection (excluding gblinear for ONNX compatibility)
        booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
        params["booster"] = booster

        # Tree-specific parameters
        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        params["grow_policy"] = grow_policy

        if grow_policy == "lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 0, 256)

        if params["tree_method"] == "hist":
            params["max_bin"] = trial.suggest_int("max_bin", 128, 512)

        # Dart-specific parameters
        if booster == "dart":
            params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            params["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.3)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.5)

        # Task-specific parameters
        params = self._add_task_specific_params(trial, params)

        return params

    def _add_task_specific_params(self, trial, params: Dict) -> Dict:
        """Add task-specific parameters."""
        if self.problem_type == "binary_classification":
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 10.0, log=True)
            params["max_delta_step"] = trial.suggest_float("max_delta_step", 0.0, 10.0)

        elif self.problem_type == "multi_class_classification":
            params["max_depth"] = trial.suggest_int("max_depth", 4, 15)

        elif self.problem_type == "multi_label_classification":
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 0.1, 10.0, log=True)
            params["max_delta_step"] = trial.suggest_float("max_delta_step", 0.0, 10.0)

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            # Note: reg:squaredlogerror excluded as it requires labels > -1
            objective = trial.suggest_categorical(
                "objective",
                ["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"],
            )
            params["objective"] = objective
            if objective == "reg:pseudohubererror":
                params["huber_slope"] = trial.suggest_float("huber_slope", 0.5, 2.0)

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
        """Train XGBoost model."""
        # Extract early stopping rounds
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)

        # Create model based on problem type
        if self.problem_type in ("binary_classification", "multi_class_classification"):
            self.model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        else:
            self.model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)

        # Fit model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        self.best_iteration = self.model.best_iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        """Get the underlying XGBoost model."""
        return self.model

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

    def get_booster(self):
        """Get the XGBoost booster for ONNX export."""
        return self.model.get_booster()
