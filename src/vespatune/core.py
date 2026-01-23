import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from .enums import ProblemType
from .export import export_model
from .logger import logger
from .models import MODEL_REGISTRY, get_preprocessor, list_models
from .schemas import ModelConfig
from .utils import is_cuda_available, reduce_memory_usage, train_final_model, train_model


@dataclass
class VespaTune:
    # required arguments
    train_filename: str
    valid_filename: str
    output: str

    # optional arguments
    test_filename: Optional[str] = None
    task: Optional[str] = None
    idx: Optional[str] = "id"
    targets: Optional[List[str]] = None
    features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    use_gpu: Optional[bool] = None  # Auto-detect if None
    seed: Optional[int] = 42
    num_trials: Optional[int] = 1000
    time_limit: Optional[int] = None
    model_type: Optional[str] = "xgboost"  # xgboost, lightgbm, or catboost

    def __post_init__(self):
        os.makedirs(self.output, exist_ok=True)
        logger.info(f"Output directory: {self.output}")

        if self.targets is None:
            logger.warning("No target columns specified. Will default to `target`.")
            self.targets = ["target"]

        if self.idx is None:
            logger.warning("No id column specified. Will default to `id`.")
            self.idx = "id"

        # Validate model type
        available_models = list_models()
        if self.model_type.lower() not in available_models:
            raise ValueError(
                f"Unknown model type: {self.model_type}. " f"Available models: {', '.join(available_models)}"
            )
        self.model_type = self.model_type.lower()
        logger.info(f"Model type: {self.model_type}")

        # Auto-detect GPU if not explicitly set
        if self.use_gpu is None:
            self.use_gpu = is_cuda_available()
        if self.use_gpu:
            logger.info("GPU training enabled")

    def _determine_problem_type(self, train_df):
        if self.task is not None:
            if self.task == "classification":
                if len(self.targets) == 1:
                    if len(np.unique(train_df[self.targets].values)) == 2:
                        problem_type = ProblemType.binary_classification
                    else:
                        problem_type = ProblemType.multi_class_classification
                else:
                    problem_type = ProblemType.multi_label_classification

            elif self.task == "regression":
                if len(self.targets) == 1:
                    problem_type = ProblemType.single_column_regression
                else:
                    problem_type = ProblemType.multi_column_regression
            else:
                raise Exception("Problem type not understood")

        else:
            target_type = type_of_target(train_df[self.targets].values)
            if target_type == "continuous":
                problem_type = ProblemType.single_column_regression
            elif target_type == "continuous-multioutput":
                problem_type = ProblemType.multi_column_regression
            elif target_type == "binary":
                problem_type = ProblemType.binary_classification
            elif target_type == "multiclass":
                problem_type = ProblemType.multi_class_classification
            elif target_type == "multilabel-indicator":
                problem_type = ProblemType.multi_label_classification
            else:
                raise Exception("Unable to infer `problem_type`. Please provide `classification` or `regression`")
        logger.info(f"Problem type: {problem_type.name}")
        return problem_type

    def _inject_idx(self, df):
        if self.idx not in df.columns:
            df[self.idx] = np.arange(len(df))
        return df

    def _process_data(self):
        logger.info("Reading training data")
        train_df = pd.read_csv(self.train_filename)
        train_df = reduce_memory_usage(train_df)

        logger.info("Reading validation data")
        valid_df = pd.read_csv(self.valid_filename)
        valid_df = reduce_memory_usage(valid_df)

        problem_type = self._determine_problem_type(train_df)

        train_df = self._inject_idx(train_df)
        valid_df = self._inject_idx(valid_df)

        if self.test_filename is not None:
            logger.info("Reading test data")
            test_df = pd.read_csv(self.test_filename)
            test_df = reduce_memory_usage(test_df)
            test_df = self._inject_idx(test_df)

        ignore_columns = [self.idx] + self.targets

        if self.features is None:
            self.features = list(train_df.columns)
            self.features = [x for x in self.features if x not in ignore_columns]

        # Check if model searches over preprocessing parameters
        model_class = MODEL_REGISTRY[self.model_type]
        searches_preprocessing = getattr(model_class, "searches_preprocessing", False)

        # Create and fit model-specific preprocessor
        logger.info(f"Fitting {self.model_type} preprocessor")
        preprocessor = get_preprocessor(
            self.model_type,
            categorical_features=self.categorical_features,
            features=self.features,
        )
        preprocessor.fit(
            train_df,
            problem_type=problem_type,
            targets=self.targets,
            idx=self.idx,
        )

        # Get the actual categorical features (may be auto-detected)
        categorical_features = preprocessor.categorical_features_

        if searches_preprocessing:
            # For models that search preprocessing, save raw data
            # Only transform targets, preprocessing happens per trial
            logger.info("Model searches preprocessing - saving raw data")

            # Transform targets for classification
            if preprocessor.target_encoder_ is not None:
                for target_col in self.targets:
                    train_df[target_col] = preprocessor.transform_target(train_df[target_col].values)
                    valid_df[target_col] = preprocessor.transform_target(valid_df[target_col].values)

            # Save target encoder separately for later use
            joblib.dump(preprocessor.target_encoder_, f"{self.output}/vtune.target_encoder")

            # save raw data
            train_df.to_feather(os.path.join(self.output, "train.feather"))
            valid_df.to_feather(os.path.join(self.output, "valid.feather"))
            if self.test_filename is not None:
                test_df.to_feather(os.path.join(self.output, "test.feather"))

        else:
            # For models that don't search preprocessing, transform once
            # Transform features (in-place update of DataFrames)
            train_transformed = preprocessor.transform(train_df)
            valid_transformed = preprocessor.transform(valid_df)

            # Get feature names in transformed order
            feature_names_out = preprocessor.get_feature_names_out()

            # Update DataFrames with transformed features
            for i, col in enumerate(feature_names_out):
                train_df[col] = train_transformed[:, i]
                valid_df[col] = valid_transformed[:, i]

            if self.test_filename is not None:
                test_transformed = preprocessor.transform(test_df)
                for i, col in enumerate(feature_names_out):
                    test_df[col] = test_transformed[:, i]

            # Transform targets for classification
            if preprocessor.target_encoder_ is not None:
                for target_col in self.targets:
                    train_df[target_col] = preprocessor.transform_target(train_df[target_col].values)
                    valid_df[target_col] = preprocessor.transform_target(valid_df[target_col].values)

            # Update features to use the transformed order
            self.features = feature_names_out

            # save processed data
            train_df.to_feather(os.path.join(self.output, "train.feather"))
            valid_df.to_feather(os.path.join(self.output, "valid.feather"))
            if self.test_filename is not None:
                test_df.to_feather(os.path.join(self.output, "test.feather"))

            # Save preprocessor
            logger.info("Saving preprocessor")
            preprocessor.save(f"{self.output}/vtune.preprocessor")

            # Save target encoder separately (for test predictions)
            joblib.dump(preprocessor.target_encoder_, f"{self.output}/vtune.target_encoder")

        # save config
        model_config = {
            "idx": self.idx,
            "features": self.features,
            "categorical_features": categorical_features,
            "train_filename": self.train_filename,
            "valid_filename": self.valid_filename,
            "test_filename": self.test_filename,
            "output": self.output,
            "problem_type": problem_type,
            "targets": self.targets,
            "use_gpu": self.use_gpu,
            "seed": self.seed,
            "num_trials": self.num_trials,
            "time_limit": self.time_limit,
            "model_type": self.model_type,
            "searches_preprocessing": searches_preprocessing,
        }

        self.model_config = ModelConfig(**model_config)
        logger.info(f"Model config: {self.model_config}")
        logger.info("Saving model config")
        joblib.dump(self.model_config, f"{self.output}/vtune.config")

    def train(self, callbacks=None):
        self._process_data()
        best_params = train_model(self.model_config, callbacks=callbacks)
        logger.info("Hyperparameter tuning complete")

        # Save best params for later use
        joblib.dump(best_params, f"{self.output}/vtune.best_params")

        # Train final model on all data (train + valid)
        self.train_final(best_params)

    def train_final(self, best_params):
        logger.info("Training final model on all data")
        train_final_model(self.model_config, best_params)

        # Export to ONNX
        logger.info("Exporting model to ONNX format")
        try:
            export_model(self.output)
            logger.info("ONNX export complete")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
