import json
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import create_model

from .enums import ProblemType
from .logger import logger
from .preprocessor import BasePreprocessor
from .utils import reduce_memory_usage, use_predict_proba

xgb.set_config(verbosity=0)


@dataclass
class VespaTuneProcessor:
    """Standalone preprocessor for VespaTune models.

    Use this when you want to preprocess data independently and pass it
    to an ONNX model or external inference system.

    Example:
        >>> processor = VespaTuneProcessor(model_path="./output")
        >>> # For DataFrame input
        >>> processed = processor.transform(df)
        >>> # For single sample
        >>> processed = processor.transform_single({"feature1": 1.0, "feature2": "A"})
        >>> # Then pass to ONNX runtime
        >>> session.run(None, {"input": processed})
    """

    model_path: str

    def __post_init__(self):
        # Try loading from ONNX export directory first (has metadata.json)
        metadata_path = os.path.join(self.model_path, "metadata.json")
        if os.path.exists(metadata_path):
            # ONNX export directory
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.features = metadata["features"]
            self.categorical_features = metadata["categorical_features"]
            self.targets = metadata["targets"]
            self.problem_type = metadata["problem_type"]
            self.idx = metadata.get("idx", "id")
            preprocessor_path = os.path.join(self.model_path, "preprocessor.joblib")
        else:
            # Regular model directory
            config_path = os.path.join(self.model_path, "vtune.config")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Neither metadata.json nor vtune.config found in {self.model_path}")
            model_config = joblib.load(config_path)
            self.features = model_config.features
            self.categorical_features = model_config.categorical_features
            self.targets = model_config.targets
            self.problem_type = model_config.problem_type.name
            self.idx = model_config.idx
            preprocessor_path = os.path.join(self.model_path, "vtune.preprocessor.joblib")

        self.preprocessor = BasePreprocessor.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")

    def get_feature_names(self) -> List[str]:
        """Get the list of input feature names."""
        return self.features

    def get_categorical_features(self) -> List[str]:
        """Get the list of categorical feature names."""
        return self.categorical_features

    def get_feature_names_out(self) -> List[str]:
        """Get the list of output feature names after transformation."""
        return self.preprocessor.get_feature_names_out()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame to model-ready features.

        Args:
            df: Input DataFrame with raw features.

        Returns:
            Numpy array of transformed features ready for model inference.
        """
        return self.preprocessor.transform(df).astype(np.float32)

    def transform_single(self, sample: Dict[str, Union[str, int, float]]) -> np.ndarray:
        """Transform a single sample to model-ready features.

        Args:
            sample: Dictionary mapping feature names to values.

        Returns:
            Numpy array of shape (1, n_features) ready for model inference.
        """
        sample_df = pd.DataFrame.from_dict(sample, orient="index").T
        return self.transform(sample_df)

    def get_input_schema(self):
        """Generate Pydantic schema for input validation.

        Returns:
            A Pydantic model class for validating input data.
        """
        schema = {}
        for feat in self.features:
            if feat in self.categorical_features:
                schema[feat] = (str, ...)
            else:
                schema[feat] = (float, ...)
        return create_model("InputSchema", **schema)


@dataclass
class VespaTunePredict:
    model_path: str

    def __post_init__(self):
        self.model_config = joblib.load(os.path.join(self.model_path, "vtune.config"))
        self.model = joblib.load(os.path.join(self.model_path, "vtune_model.final"))
        self.use_predict_proba = use_predict_proba(self.model_config.problem_type)

        # Load preprocessor
        preprocessor_path = os.path.join(self.model_path, "vtune.preprocessor.joblib")
        self.preprocessor = BasePreprocessor.load(preprocessor_path)
        self.target_encoder = self.preprocessor.target_encoder_

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
        test_ids = df[self.model_config.idx].values

        # Preprocess features
        test_features = self.preprocessor.transform(df)

        if self.model_config.problem_type in (
            ProblemType.multi_column_regression,
            ProblemType.multi_label_classification,
        ):
            preds_list = []
            for idx in range(len(self.model)):
                if self.model_config.problem_type == ProblemType.multi_column_regression:
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
            final_preds = pd.DataFrame(final_preds, columns=list(self.target_encoder.classes_))
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
            raise ImportError("onnxruntime is required for ONNX inference. " "Install with: pip install onnxruntime")

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

        # Load preprocessor
        preprocessor_path = os.path.join(self.model_path, "preprocessor.joblib")
        self.preprocessor = BasePreprocessor.load(preprocessor_path)
        self.target_encoder = self.preprocessor.target_encoder_

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
        return self.preprocessor.transform(df)

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
