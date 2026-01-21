import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import create_model

from .enums import ProblemType
from .logger import logger
from .utils import fetch_xgb_model_params, reduce_memory_usage


xgb.set_config(verbosity=0)


@dataclass
class VespaTunePredict:
    model_path: str

    def __post_init__(self):
        self.model_config = joblib.load(os.path.join(self.model_path, "vtune.config"))
        self.target_encoder = joblib.load(
            os.path.join(self.model_path, "vtune.target_encoder")
        )
        self.categorical_encoder = joblib.load(
            os.path.join(self.model_path, "vtune.categorical_encoder")
        )
        self.model = joblib.load(os.path.join(self.model_path, "vtune_model.final"))
        _, self.use_predict_proba, _, _ = fetch_xgb_model_params(self.model_config)

    def get_prediction_schema(self):
        cat_features = self.model_config.categorical_features
        schema = {"PredictSchema": {}}
        for cf in cat_features:
            schema["PredictSchema"][cf] = (str, ...)

        for feat in self.model_config.features:
            if feat not in cat_features:
                schema["PredictSchema"][feat] = (float, ...)
        return create_model("PredictSchema", **schema["PredictSchema"])

    def _predict_df(self, df):
        categorical_features = self.model_config.categorical_features
        test_ids = df[self.model_config.idx].values

        test_df = df.copy(deep=True)
        if len(categorical_features) > 0 and self.categorical_encoder is not None:
            test_df[categorical_features] = self.categorical_encoder.transform(
                test_df[categorical_features].values
            )

        test_features = test_df[self.model_config.features]

        for col in test_features.columns:
            if test_features[col].dtype == "object":
                test_features[col] = test_features[col].astype(np.int64)

        if self.model_config.problem_type in (
            ProblemType.multi_column_regression,
            ProblemType.multi_label_classification,
        ):
            preds_list = []
            for idx in range(len(self.model)):
                if (
                    self.model_config.problem_type
                    == ProblemType.multi_column_regression
                ):
                    pred = self.model[idx].predict(test_features)
                else:
                    pred = self.model[idx].predict_proba(test_features)[:, 1]
                preds_list.append(pred)
            final_preds = np.column_stack(preds_list)
        else:
            if self.use_predict_proba:
                final_preds = self.model.predict_proba(test_features)
            else:
                final_preds = self.model.predict(test_features)

        if self.target_encoder is None:
            final_preds = pd.DataFrame(final_preds, columns=self.model_config.targets)
        else:
            final_preds = pd.DataFrame(
                final_preds, columns=list(self.target_encoder.classes_)
            )
        final_preds.insert(loc=0, column=self.model_config.idx, value=test_ids)
        return final_preds

    def predict_single(self, sample: Dict[str, Union[str, int, float]] = None):
        sample_df = pd.DataFrame.from_dict(sample, orient="index").T
        sample_df[self.model_config.idx] = 0
        preds = self._predict_df(sample_df)
        preds = preds.to_dict(orient="records")[0]
        return preds

    def predict_file(self, test_filename: str, out_filename: str):
        test_df = pd.read_csv(test_filename)
        test_df = reduce_memory_usage(test_df)
        if self.model_config.idx not in test_df.columns:
            test_df[self.model_config.idx] = np.arange(len(test_df))
        final_preds = self._predict_df(test_df)
        final_preds.to_csv(out_filename, index=False)


@dataclass
class VespaTuneONNXPredict:
    """ONNX-based predictor for VespaTune models.

    This predictor uses ONNX Runtime for inference, making it model-agnostic
    and suitable for production deployments.
    """

    model_path: str

    def __post_init__(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install onnxruntime"
            )

        # Load metadata
        metadata_path = os.path.join(self.model_path, "metadata.json")
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.feature_mapping = self.metadata["feature_mapping"]
        self.targets = self.metadata["targets"]
        self.problem_type = self.metadata["problem_type"]
        self.categorical_features = self.metadata["categorical_features"]
        self.idx = self.metadata.get("idx", "id")

        # Load encoders
        cat_encoder_path = os.path.join(self.model_path, "categorical_encoder.joblib")
        if os.path.exists(cat_encoder_path):
            self.categorical_encoder = joblib.load(cat_encoder_path)
        else:
            self.categorical_encoder = None

        target_encoder_path = os.path.join(self.model_path, "target_encoder.joblib")
        if os.path.exists(target_encoder_path):
            self.target_encoder = joblib.load(target_encoder_path)
        else:
            self.target_encoder = None

        # Load ONNX model(s)
        self.sessions = self._load_onnx_sessions(ort)

        # Determine if we should use probabilities
        self.use_predict_proba = self.problem_type in (
            "binary_classification",
            "multi_class_classification",
            "multi_label_classification",
        )

        logger.info(f"Loaded ONNX model from {self.model_path}")

    def _load_onnx_sessions(self, ort) -> List:
        """Load ONNX inference sessions."""
        sessions = []

        if self.problem_type in ("multi_column_regression", "multi_label_classification"):
            # Multiple models, one per target
            for target in self.targets:
                model_path = os.path.join(self.model_path, f"model_{target}.onnx")
                session = ort.InferenceSession(model_path)
                sessions.append(session)
        else:
            # Single model
            model_path = os.path.join(self.model_path, "model.onnx")
            session = ort.InferenceSession(model_path)
            sessions.append(session)

        return sessions

    def get_prediction_schema(self):
        """Generate Pydantic schema for API input validation."""
        schema = {}
        for feat in self.features:
            if feat in self.categorical_features:
                schema[feat] = (str, ...)
            else:
                schema[feat] = (float, ...)
        return create_model("PredictSchema", **schema)

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess input data for ONNX inference."""
        test_df = df.copy(deep=True)

        # Encode categorical features
        if len(self.categorical_features) > 0 and self.categorical_encoder is not None:
            test_df[self.categorical_features] = self.categorical_encoder.transform(
                test_df[self.categorical_features].values
            )

        # Extract features in correct order
        features_array = test_df[self.features].values.astype(np.float32)
        return features_array

    def _run_inference(self, features: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if self.problem_type in ("multi_column_regression", "multi_label_classification"):
            preds_list = []
            for i, session in enumerate(self.sessions):
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: features})

                if self.problem_type == "multi_label_classification":
                    # Get probability of positive class
                    pred = outputs[1][:, 1] if len(outputs) > 1 else outputs[0]
                else:
                    pred = outputs[0]

                preds_list.append(pred.flatten())
            return np.column_stack(preds_list)
        else:
            session = self.sessions[0]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: features})

            if self.use_predict_proba and len(outputs) > 1:
                # Classification: return probabilities
                return outputs[1]
            else:
                return outputs[0]

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on a DataFrame."""
        test_ids = df[self.idx].values if self.idx in df.columns else np.arange(len(df))

        features = self._preprocess(df)
        predictions = self._run_inference(features)

        # Format output based on problem type and prediction shape
        if self.problem_type in ("binary_classification", "multi_class_classification"):
            # Classification returns probabilities for each class
            if self.target_encoder is not None:
                columns = list(self.target_encoder.classes_)
            else:
                # Generate class labels based on prediction shape
                n_classes = predictions.shape[1] if predictions.ndim > 1 else 2
                columns = [f"class_{i}" for i in range(n_classes)]
        else:
            # Regression
            columns = self.targets

        # Ensure predictions are 2D for DataFrame creation
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        final_preds = pd.DataFrame(predictions, columns=columns)
        final_preds.insert(loc=0, column=self.idx, value=test_ids)
        return final_preds

    def predict_single(self, sample: Dict[str, Union[str, int, float]]) -> Dict:
        """Make prediction for a single sample."""
        sample_df = pd.DataFrame.from_dict(sample, orient="index").T
        sample_df[self.idx] = 0
        preds = self._predict_df(sample_df)
        return preds.to_dict(orient="records")[0]

    def predict_file(self, test_filename: str, out_filename: str):
        """Make predictions on a file and save results."""
        test_df = pd.read_csv(test_filename)
        test_df = reduce_memory_usage(test_df)
        if self.idx not in test_df.columns:
            test_df[self.idx] = np.arange(len(test_df))
        final_preds = self._predict_df(test_df)
        final_preds.to_csv(out_filename, index=False)
