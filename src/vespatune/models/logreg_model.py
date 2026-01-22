from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

from ..preprocessor import BasePreprocessor
from .base import BaseModel


class LogRegPreprocessor(BasePreprocessor):
    """Preprocessor for Logistic Regression models.

    Uses one-hot encoding for categorical features (required for linear models).
    Defaults to standard scaling since logistic regression benefits from scaled features.
    """

    unknown_value = np.nan
    encoding = "onehot"  # One-hot encoding required for linear models

    def __init__(
        self,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_impute_strategy: Optional[str] = "median",
        categorical_impute_strategy: Optional[str] = "most_frequent",
        scaler: Optional[str] = "standard",
    ):
        super().__init__(
            features=features,
            categorical_features=categorical_features,
            numeric_impute_strategy=numeric_impute_strategy,
            categorical_impute_strategy=categorical_impute_strategy,
            scaler=scaler,
        )


class LogRegModel(BaseModel):
    """Logistic Regression model implementation using sklearn.

    Only supports classification tasks (binary, multi-class, multi-label).
    """

    name = "logreg"
    supports_categorical = False  # Requires encoding
    supports_gpu = False
    searches_preprocessing = True  # Preprocessing is part of HP search
    supported_problem_types = [
        "binary_classification",
        "multi_class_classification",
        "multi_label_classification",
    ]

    def __init__(self, problem_type: str, random_state: int = 42):
        super().__init__(problem_type, random_state)
        self.model = None

    def get_params(self, trial, model_config) -> Dict[str, Any]:
        """Get Logistic Regression hyperparameters for Optuna trial.

        Includes both model hyperparameters and preprocessing hyperparameters.
        Uses sklearn 1.8+ API with l1_ratio instead of penalty parameter.
        """
        # Preprocessing hyperparameters (encoding is fixed as class attribute)
        preprocessing_params = {
            "numeric_impute_strategy": trial.suggest_categorical("numeric_impute_strategy", ["mean", "median", None]),
            "categorical_impute_strategy": trial.suggest_categorical(
                "categorical_impute_strategy", ["most_frequent", None]
            ),
            "scaler": trial.suggest_categorical("scaler", ["standard", "minmax", "robust", None]),
        }

        # Model hyperparameters (sklearn 1.8+ API)
        # l1_ratio: 0 = L2, 1 = L1, between = elasticnet
        # C: regularization strength (np.inf = no regularization)
        use_regularization = trial.suggest_categorical("use_regularization", [True, False])

        model_params = {
            "random_state": self.random_state,
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
        }

        if use_regularization:
            # With regularization: search over l1_ratio and C
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            model_params["l1_ratio"] = l1_ratio
            model_params["C"] = trial.suggest_float("C", 1e-4, 100.0, log=True)

            # Solver selection based on l1_ratio
            if l1_ratio == 0:  # Pure L2
                solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
            elif l1_ratio == 1:  # Pure L1
                solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
            else:  # Elasticnet (0 < l1_ratio < 1)
                solver = "saga"  # Only saga supports elasticnet
            model_params["solver"] = solver
        else:
            # No regularization
            model_params["C"] = np.inf
            solver = trial.suggest_categorical("solver_none", ["lbfgs", "newton-cg", "sag", "saga"])
            model_params["solver"] = solver

        # Class weight for imbalanced data
        if self.problem_type in ("binary_classification", "multi_class_classification"):
            class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
            model_params["class_weight"] = class_weight

        # Combine preprocessing and model params
        return {
            "preprocessing": preprocessing_params,
            "model": model_params,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        params: Dict[str, Any],
        categorical_features: Optional[List[int]] = None,
    ) -> None:
        """Train Logistic Regression model."""
        # Extract model params (preprocessing is handled separately)
        model_params = params.get("model", params).copy()

        # Filter out non-sklearn parameters
        non_sklearn_params = {
            # HP search control params
            "use_regularization",
            "solver_l2",
            "solver_l1",
            "solver_none",
            # Preprocessing params (handled separately)
            "numeric_impute_strategy",
            "categorical_impute_strategy",
            "scaler",
        }
        for param in non_sklearn_params:
            model_params.pop(param, None)

        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train, y_train)

        # LogReg doesn't have early stopping, but we can track validation score
        self.best_iteration = 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_model(self) -> Any:
        """Get the underlying sklearn model."""
        return self.model

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(path)

    def to_onnx(self, n_features: int) -> onnx.ModelProto:
        """Convert Logistic Regression model to ONNX format."""
        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(
            self.model,
            initial_types=initial_types,
            target_opset=15,
        )
        return onnx_model
