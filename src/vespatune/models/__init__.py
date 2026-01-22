"""Model implementations for VespaTune.

To add a new model:
1. Create a new file in this directory (e.g., newmodel_model.py)
2. Implement a class that inherits from BaseModel
3. Implement a Preprocessor class that inherits from BasePreprocessor
4. Add the model and preprocessor to the registries below
"""

from typing import Dict, Type

from ..preprocessor import BasePreprocessor
from .base import BaseModel
from .catboost_model import CatBoostModel, CatBoostPreprocessor
from .lightgbm_model import LightGBMModel, LightGBMPreprocessor
from .logreg_model import LogRegModel, LogRegPreprocessor
from .xgboost_model import XGBoostModel, XGBoostPreprocessor


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "logreg": LogRegModel,
}

# Registry of model-specific preprocessors
PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessor]] = {
    "xgboost": XGBoostPreprocessor,
    "lightgbm": LightGBMPreprocessor,
    "catboost": CatBoostPreprocessor,
    "logreg": LogRegPreprocessor,
}

# Default model
DEFAULT_MODEL = "xgboost"


def get_model(model_name: str, problem_type: str, random_state: int = 42) -> BaseModel:
    """Get a model instance by name.

    Args:
        model_name: Name of the model (xgboost, lightgbm, catboost)
        problem_type: Type of problem (binary_classification, etc.)
        random_state: Random seed for reproducibility

    Returns:
        Model instance

    Raises:
        ValueError: If model_name is not registered
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(problem_type=problem_type, random_state=random_state)


def list_models() -> list:
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """Register a new model.

    Args:
        name: Name to register the model under
        model_class: Model class (must inherit from BaseModel)
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"{model_class} must inherit from BaseModel")
    MODEL_REGISTRY[name.lower()] = model_class


def get_preprocessor(model_name: str, **kwargs) -> BasePreprocessor:
    """Get a preprocessor instance for a specific model type.

    Args:
        model_name: Name of the model (xgboost, lightgbm, catboost)
        **kwargs: Arguments passed to the preprocessor constructor

    Returns:
        Preprocessor instance for the specified model

    Raises:
        ValueError: If model_name is not registered
    """
    model_name = model_name.lower()
    if model_name not in PREPROCESSOR_REGISTRY:
        available = ", ".join(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    preprocessor_class = PREPROCESSOR_REGISTRY[model_name]
    return preprocessor_class(**kwargs)
