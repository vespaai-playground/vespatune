"""Model implementations for VespaTune.

To add a new model:
1. Create a new file in this directory (e.g., newmodel_model.py)
2. Implement a class that inherits from BaseModel
3. Add the model to MODEL_REGISTRY below
"""

from typing import Dict, Type

from .base import BaseModel
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
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
