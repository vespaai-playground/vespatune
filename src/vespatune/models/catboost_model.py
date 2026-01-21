from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost model implementation."""

    name = "catboost"
    supports_categorical = True  # Best native categorical support
    supports_gpu = True

    def __init__(self, problem_type: str, random_state: int = 42):
        super().__init__(problem_type, random_state)
        self.model = None

    def get_params(self, trial, model_config) -> Dict[str, Any]:
        """Get CatBoost hyperparameters for Optuna trial."""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 500, 10000),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 300),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_state": self.random_state,
            "verbose": False,
        }

        # Bootstrap type
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        params["bootstrap_type"] = bootstrap_type

        if bootstrap_type == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        elif bootstrap_type == "MVS":
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        # Grow policy
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
        params["grow_policy"] = grow_policy

        if grow_policy in ("Depthwise", "Lossguide"):
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 100)

        if grow_policy == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

        # Feature sampling
        params["rsm"] = trial.suggest_float("rsm", 0.3, 1.0)  # colsample_bylevel equivalent

        # GPU settings
        if model_config.use_gpu:
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"

        # Leaf estimation
        params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)

        # Task-specific parameters
        params = self._add_task_specific_params(trial, params)

        return params

    def _add_task_specific_params(self, trial, params: Dict) -> Dict:
        """Add task-specific parameters."""
        if self.problem_type == "binary_classification":
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "Logloss"
            params["auto_class_weights"] = trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced", "SqrtBalanced"]
            )

        elif self.problem_type == "multi_class_classification":
            params["loss_function"] = "MultiClass"
            params["eval_metric"] = "MultiClass"
            params["auto_class_weights"] = trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced", "SqrtBalanced"]
            )

        elif self.problem_type == "multi_label_classification":
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "Logloss"

        elif self.problem_type in ("single_column_regression", "multi_column_regression"):
            loss_function = trial.suggest_categorical(
                "loss_function", ["RMSE", "MAE", "Quantile", "Huber", "LogLinQuantile"]
            )
            params["loss_function"] = loss_function
            params["eval_metric"] = "RMSE"

            if loss_function == "Quantile":
                params["quantile"] = trial.suggest_float("quantile", 0.1, 0.9)
            elif loss_function == "Huber":
                params["huber_delta"] = trial.suggest_float("huber_delta", 0.5, 2.0)

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
        """Train CatBoost model."""
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        # Extract early stopping rounds
        early_stopping_rounds = params.pop("early_stopping_rounds", 100)
        params["early_stopping_rounds"] = early_stopping_rounds

        # Create pools with categorical features
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

        # Create model based on problem type
        if self.problem_type in ("binary_classification", "multi_class_classification"):
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        # Fit model
        self.model.fit(train_pool, eval_set=valid_pool, verbose=False)

        self.best_iteration = self.model.best_iteration_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        """Get the underlying CatBoost model."""
        return self.model

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

    def save_native(self, path: str) -> None:
        """Save model in CatBoost native format."""
        self.model.save_model(path)

    def load_native(self, path: str) -> None:
        """Load model from CatBoost native format."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        if self.problem_type in ("binary_classification", "multi_class_classification"):
            self.model = CatBoostClassifier()
        else:
            self.model = CatBoostRegressor()
        self.model.load_model(path)
