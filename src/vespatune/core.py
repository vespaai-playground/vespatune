import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.multiclass import type_of_target

from .enums import ProblemType
from .logger import logger
from .schemas import ModelConfig
from .models import list_models
from .utils import reduce_memory_usage, train_final_model, train_model


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
    use_gpu: Optional[bool] = False
    seed: Optional[int] = 42
    num_trials: Optional[int] = 1000
    time_limit: Optional[int] = None
    model_type: Optional[str] = "xgboost"  # xgboost, lightgbm, or catboost

    def __post_init__(self):
        if os.path.exists(self.output):
            raise Exception(
                "Output directory already exists. Please specify some other directory."
            )
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
                f"Unknown model type: {self.model_type}. "
                f"Available models: {', '.join(available_models)}"
            )
        self.model_type = self.model_type.lower()
        logger.info(f"Model type: {self.model_type}")

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
                raise Exception(
                    "Unable to infer `problem_type`. Please provide `classification` or `regression`"
                )
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

        # encode target(s)
        if problem_type in [
            ProblemType.binary_classification,
            ProblemType.multi_class_classification,
        ]:
            logger.info("Encoding target(s)")
            target_encoder = LabelEncoder()
            target_encoder.fit(train_df[self.targets].values.reshape(-1))
            train_df.loc[:, self.targets] = target_encoder.transform(
                train_df[self.targets].values.reshape(-1)
            )
            valid_df.loc[:, self.targets] = target_encoder.transform(
                valid_df[self.targets].values.reshape(-1)
            )
        else:
            target_encoder = None

        if self.categorical_features is None:
            categorical_features = []
            for col in self.features:
                if train_df[col].dtype == "object":
                    categorical_features.append(col)
        else:
            categorical_features = self.categorical_features

        logger.info(f"Found {len(categorical_features)} categorical features.")

        # encode categorical features
        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")
            categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )
            train_df[categorical_features] = categorical_encoder.fit_transform(
                train_df[categorical_features].values
            )
            valid_df[categorical_features] = categorical_encoder.transform(
                valid_df[categorical_features].values
            )
            if self.test_filename is not None:
                test_df[categorical_features] = categorical_encoder.transform(
                    test_df[categorical_features].values
                )
        else:
            categorical_encoder = None

        # save processed data
        train_df.to_feather(os.path.join(self.output, "train.feather"))
        valid_df.to_feather(os.path.join(self.output, "valid.feather"))
        if self.test_filename is not None:
            test_df.to_feather(os.path.join(self.output, "test.feather"))

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
        }

        self.model_config = ModelConfig(**model_config)
        logger.info(f"Model config: {self.model_config}")
        logger.info("Saving model config")
        joblib.dump(self.model_config, f"{self.output}/vtune.config")

        # save encoders
        logger.info("Saving encoders")
        joblib.dump(categorical_encoder, f"{self.output}/vtune.categorical_encoder")
        joblib.dump(target_encoder, f"{self.output}/vtune.target_encoder")

    def train(self):
        self._process_data()
        best_params = train_model(self.model_config)
        logger.info("Hyperparameter tuning complete")

        # Save best params for later use
        joblib.dump(best_params, f"{self.output}/vtune.best_params")

        # Train final model on all data (train + valid)
        self.train_final(best_params)

    def train_final(self, best_params):
        logger.info("Training final model on all data")
        train_final_model(self.model_config, best_params)
