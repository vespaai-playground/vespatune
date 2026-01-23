"""Base preprocessing module for VespaTune.

Uses sklearn's ColumnTransformer internally. Model-specific preprocessors
inherit from BasePreprocessor and only override the settings they need.
"""

from typing import List, Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .enums import ProblemType
from .logger import logger


class BasePreprocessor:
    """Base preprocessor using sklearn ColumnTransformer.

    Subclasses only need to set `unknown_value` for model-specific behavior.

    Parameters
    ----------
    features : list of str, optional
        List of feature column names to use.
    categorical_features : list of str, optional
        List of categorical feature column names.
    numeric_impute_strategy : str, optional
        Imputation strategy for numeric features: "mean", "median", "most_frequent", or None.
    categorical_impute_strategy : str, optional
        Imputation strategy for categorical features: "most_frequent" or None.
    scaler : str, optional
        Scaling method for numeric features: "standard", "minmax", "robust", or None.
    """

    # Subclasses override these for model-specific behavior
    unknown_value: float = np.nan
    encoding: str = "ordinal"  # "ordinal" for tree models, "onehot" for linear models

    def __init__(
        self,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        numeric_impute_strategy: Optional[Literal["mean", "median", "most_frequent"]] = None,
        categorical_impute_strategy: Optional[Literal["most_frequent"]] = None,
        scaler: Optional[Literal["standard", "minmax", "robust"]] = None,
    ):
        self.features = features
        self.categorical_features = categorical_features
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.scaler = scaler

        # Attributes set during fit
        self.features_: Optional[List[str]] = None
        self.categorical_features_: Optional[List[str]] = None
        self.numeric_features_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.problem_type_: Optional[ProblemType] = None
        self.target_encoder_: Optional[LabelEncoder] = None
        self.pipeline_: Optional[ColumnTransformer] = None
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        problem_type: Optional[ProblemType] = None,
        targets: Optional[List[str]] = None,
        idx: Optional[str] = None,
    ) -> "BasePreprocessor":
        """Fit the preprocessor on training data."""
        self.problem_type_ = problem_type

        # Determine feature columns
        ignore_columns = []
        if idx is not None:
            ignore_columns.append(idx)
        if targets is not None:
            ignore_columns.extend(targets)

        if self.features is not None:
            self.features_ = self.features.copy()
        else:
            self.features_ = [c for c in X.columns if c not in ignore_columns]

        # Detect categorical features if not provided
        if self.categorical_features is not None:
            self.categorical_features_ = [c for c in self.categorical_features if c in self.features_]
        else:
            self.categorical_features_ = [
                c for c in self.features_ if pd.api.types.is_string_dtype(X[c]) or X[c].dtype == "object"
            ]

        # Determine numeric features
        self.numeric_features_ = [c for c in self.features_ if c not in self.categorical_features_]

        self.n_features_in_ = len(self.features_)

        logger.info(f"Features: {len(self.features_)} total")
        logger.info(f"  Categorical: {len(self.categorical_features_)}")
        logger.info(f"  Numeric: {len(self.numeric_features_)}")

        # Build sklearn ColumnTransformer with pipelines
        transformers = []

        # Categorical pipeline: impute (optional) → encode
        if self.categorical_features_:
            cat_steps = []
            if self.categorical_impute_strategy:
                cat_steps.append(
                    (
                        "imputer",
                        SimpleImputer(strategy=self.categorical_impute_strategy),
                    )
                )

            # Choose encoder based on encoding parameter
            if self.encoding == "onehot":
                cat_steps.append(
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",  # Unknown categories become all zeros
                            sparse_output=False,
                            dtype=np.float32,
                        ),
                    )
                )
            else:  # ordinal (default)
                cat_steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=self.unknown_value,
                            dtype=np.float32,
                        ),
                    )
                )
            cat_transformer = Pipeline(cat_steps) if len(cat_steps) > 1 else cat_steps[0][1]
            transformers.append(("cat", cat_transformer, self.categorical_features_))

        # Numeric pipeline: impute (optional) → scale (optional)
        if self.numeric_features_:
            num_steps = []
            if self.numeric_impute_strategy:
                num_steps.append(
                    (
                        "imputer",
                        SimpleImputer(strategy=self.numeric_impute_strategy),
                    )
                )
            if self.scaler:
                scaler_map = {
                    "standard": StandardScaler(),
                    "minmax": MinMaxScaler(),
                    "robust": RobustScaler(),
                }
                num_steps.append(("scaler", scaler_map[self.scaler]))

            if num_steps:
                num_transformer = Pipeline(num_steps) if len(num_steps) > 1 else num_steps[0][1]
            else:
                num_transformer = "passthrough"
            transformers.append(("num", num_transformer, self.numeric_features_))

        self.pipeline_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.pipeline_.fit(X[self.features_])

        # Fit target encoder for classification
        if y is not None and problem_type is not None:
            self._fit_target_encoder(y, problem_type)
        elif targets is not None and problem_type is not None:
            y_data = X[targets].values
            if len(targets) == 1:
                y_data = y_data.ravel()
            self._fit_target_encoder(y_data, problem_type)

        self._is_fitted = True
        return self

    def _fit_target_encoder(
        self,
        y: Union[pd.Series, np.ndarray],
        problem_type: ProblemType,
    ) -> None:
        """Fit target encoder for classification problems."""
        if problem_type in (
            ProblemType.binary_classification,
            ProblemType.multi_class_classification,
        ):
            logger.info("Fitting target encoder")
            self.target_encoder_ = LabelEncoder()
            y_flat = np.asarray(y).ravel()
            self.target_encoder_.fit(y_flat)
        else:
            self.target_encoder_ = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features to float32 array."""
        self._check_is_fitted()
        result = self.pipeline_.transform(X[self.features_])
        return result.astype(np.float32)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        **fit_params,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform_target(self, y: Union[pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform target values using the fitted target encoder."""
        self._check_is_fitted()
        if self.target_encoder_ is None:
            return np.asarray(y)
        y_flat = np.asarray(y).ravel()
        return self.target_encoder_.transform(y_flat)

    def inverse_transform_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Inverse transform encoded target values."""
        self._check_is_fitted()
        if self.target_encoder_ is None:
            return np.asarray(y)
        y_flat = np.asarray(y).ravel()
        return self.target_encoder_.inverse_transform(y_flat.astype(int))

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        self._check_is_fitted()
        return list(self.pipeline_.get_feature_names_out())

    def get_categorical_indices(self) -> List[int]:
        """Get indices of categorical features in output.

        For ordinal encoding, returns indices of original categorical columns.
        For one-hot encoding, returns indices of all one-hot encoded columns.
        """
        self._check_is_fitted()
        feature_names = self.get_feature_names_out()

        if self.encoding == "onehot":
            # One-hot encoded columns have names like "cat1_value1", "cat1_value2", etc.
            # Return indices of all columns that start with any categorical feature name
            indices = []
            for i, name in enumerate(feature_names):
                for cat_col in self.categorical_features_:
                    if name.startswith(f"{cat_col}_") or name == cat_col:
                        indices.append(i)
                        break
            return indices
        else:
            # Ordinal encoding keeps original column names
            return [feature_names.index(c) for c in self.categorical_features_ if c in feature_names]

    def _check_is_fitted(self) -> None:
        """Check if the preprocessor has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

    def save(self, path: str) -> None:
        """Save the preprocessor to disk."""
        self._check_is_fitted()
        full_path = f"{path}.joblib" if not path.endswith(".joblib") else path
        joblib.dump(self, full_path)
        logger.info(f"Preprocessor saved to {full_path}")

    @classmethod
    def load(cls, path: str) -> "BasePreprocessor":
        """Load a preprocessor from disk."""
        full_path = f"{path}.joblib" if not path.endswith(".joblib") else path
        preprocessor = joblib.load(full_path)
        logger.info(f"Preprocessor loaded from {full_path}")
        return preprocessor

    def get_config(self) -> dict:
        """Get preprocessor configuration as a dictionary."""
        self._check_is_fitted()
        return {
            "features": self.features_,
            "categorical_features": self.categorical_features_,
            "numeric_features": self.numeric_features_,
            "n_features_in": self.n_features_in_,
            "problem_type": self.problem_type_.name if self.problem_type_ else None,
            "has_target_encoder": self.target_encoder_ is not None,
            "numeric_impute_strategy": self.numeric_impute_strategy,
            "categorical_impute_strategy": self.categorical_impute_strategy,
            "scaler": self.scaler,
            "encoding": self.encoding,
        }

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"{self.__class__.__name__}(n_features={self.n_features_in_}, "
                f"n_categorical={len(self.categorical_features_)}, "
                f"n_numeric={len(self.numeric_features_)}, fitted=True)"
            )
        return f"{self.__class__.__name__}(fitted=False)"
