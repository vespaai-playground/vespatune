from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx


class BaseModel(ABC):
    """Abstract base class for all model implementations.

    To add a new model type:
    1. Create a new file in the models/ directory (e.g., newmodel.py)
    2. Implement a class that inherits from BaseModel
    3. Register it in models/__init__.py using MODEL_REGISTRY
    """

    name: str = "base"
    supports_categorical: bool = False
    supports_gpu: bool = False
    searches_preprocessing: bool = False  # If True, preprocessing params are part of HP search
    supported_problem_types: list = None  # None means all types supported

    def __init__(self, problem_type: str, random_state: int = 42):
        # Validate problem type if model has restrictions
        if self.supported_problem_types is not None:
            if problem_type not in self.supported_problem_types:
                raise ValueError(
                    f"{self.__class__.__name__} does not support problem type '{problem_type}'. "
                    f"Supported types: {self.supported_problem_types}"
                )
        self.problem_type = problem_type
        self.random_state = random_state
        self.model = None
        self.best_iteration = None

    @abstractmethod
    def get_params(self, trial, model_config) -> Dict[str, Any]:
        """Get hyperparameters for Optuna trial.

        Args:
            trial: Optuna trial object
            model_config: Model configuration

        Returns:
            Dictionary of hyperparameters to try
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        params: Dict[str, Any],
        categorical_features: Optional[List[int]] = None,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            params: Model hyperparameters
            categorical_features: Indices of categorical features
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predictions (raw values for regression, labels for classification)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification).

        Args:
            X: Features to predict on

        Returns:
            Class probabilities
        """
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Get the underlying model object."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

    @abstractmethod
    def to_onnx(self, n_features: int) -> onnx.ModelProto:
        """Convert the model to ONNX format.

        Args:
            n_features: Number of input features

        Returns:
            ONNX model
        """
        pass

    def get_best_iteration(self) -> Optional[int]:
        """Get the best iteration from early stopping."""
        return self.best_iteration

    @classmethod
    def get_default_metric(cls, problem_type: str) -> Tuple[str, str]:
        """Get default evaluation metric for problem type.

        Args:
            problem_type: Type of problem

        Returns:
            Tuple of (metric_name, optimization_direction)
        """
        metric_map = {
            "binary_classification": ("logloss", "minimize"),
            "multi_class_classification": ("mlogloss", "minimize"),
            "multi_label_classification": ("logloss", "minimize"),
            "single_column_regression": ("rmse", "minimize"),
            "multi_column_regression": ("rmse", "minimize"),
        }
        return metric_map.get(problem_type, ("rmse", "minimize"))
