import json
import os
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import onnx
from onnxmltools.convert.common.data_types import FloatTensorType

from .enums import ProblemType
from .logger import logger


@dataclass
class VespaTuneExport:
    model_path: str

    def __post_init__(self):
        self.model_config = joblib.load(os.path.join(self.model_path, "vtune.config"))
        self.model = joblib.load(os.path.join(self.model_path, "vtune_model.final"))
        self.model_type = getattr(self.model_config, "model_type", "xgboost")

    def _get_initial_types(self) -> List:
        n_features = len(self.model_config.features)
        return [("input", FloatTensorType([None, n_features]))]

    def _check_xgboost_exportable(self, model):
        """Check if the XGBoost model can be exported to ONNX."""
        try:
            booster = model.get_booster()
            config = booster.save_config()

            config_dict = json.loads(config)
            booster_type = config_dict.get("learner", {}).get("gradient_booster", {}).get("name", "")

            if booster_type == "gblinear":
                raise ValueError(
                    "Cannot export gblinear models to ONNX. "
                    "Only gbtree and dart boosters are supported by onnxmltools."
                )
        except (KeyError, TypeError):
            pass  # Unable to determine booster type, try to export anyway

    def _convert_xgboost_model(self, model):
        """Convert XGBoost model to ONNX."""
        from onnxmltools import convert_xgboost

        self._check_xgboost_exportable(model)
        initial_types = self._get_initial_types()

        # onnxmltools expects feature names to follow 'f%d' pattern
        booster = model.get_booster()
        original_feature_names = booster.feature_names
        booster.feature_names = [f"f{i}" for i in range(len(self.model_config.features))]

        try:
            onnx_model = convert_xgboost(
                model,
                initial_types=initial_types,
                target_opset=15,
            )
        finally:
            booster.feature_names = original_feature_names

        return onnx_model

    def _convert_lightgbm_model(self, model):
        """Convert LightGBM model to ONNX."""
        from onnxmltools import convert_lightgbm

        initial_types = self._get_initial_types()

        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=15,
        )
        return onnx_model

    def _convert_catboost_model(self, model):
        """Convert CatBoost model to ONNX using native export."""
        import tempfile

        # CatBoost has native ONNX export
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model.save_model(
                tmp_path,
                format="onnx",
                export_parameters={
                    "onnx_domain": "ai.catboost",
                    "onnx_model_version": 1,
                },
            )
            onnx_model = onnx.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return onnx_model

    def _convert_single_model(self, model, target_name: Optional[str] = None):
        """Convert a single model to ONNX based on model type."""
        if self.model_type == "xgboost":
            return self._convert_xgboost_model(model)
        elif self.model_type == "lightgbm":
            return self._convert_lightgbm_model(model)
        elif self.model_type == "catboost":
            return self._convert_catboost_model(model)
        else:
            raise ValueError(f"ONNX export not supported for model type: {self.model_type}")

    def export_to_onnx(
        self,
        output_dir: Optional[str] = None,
        verify: bool = False,
    ) -> List[str]:
        """Export model to ONNX format."""
        if output_dir is None:
            output_dir = os.path.join(self.model_path, "onnx")

        os.makedirs(output_dir, exist_ok=True)

        exported_files = []

        if self.model_config.problem_type in (
            ProblemType.multi_column_regression,
            ProblemType.multi_label_classification,
        ):
            for idx, target in enumerate(self.model_config.targets):
                target_model = self.model[idx]
                onnx_model = self._convert_single_model(target_model, target_name=target)
                output_path = os.path.join(output_dir, f"model_{target}.onnx")
                onnx.save_model(onnx_model, output_path)
                exported_files.append(output_path)
                logger.info(f"Exported model for target '{target}' to {output_path}")
        else:
            onnx_model = self._convert_single_model(self.model)
            output_path = os.path.join(output_dir, "model.onnx")
            onnx.save_model(onnx_model, output_path)
            exported_files.append(output_path)
            logger.info(f"Exported model to {output_path}")

        self._export_metadata(output_dir)
        self._export_encoders(output_dir)

        if verify:
            self._verify_exports(exported_files)

        logger.info(f"Successfully exported {len(exported_files)} ONNX model(s) to {output_dir}")
        return exported_files

    def _export_metadata(self, output_dir: str):
        # Create mapping from ONNX feature names (f0, f1, ...) to original names
        feature_mapping = {f"f{i}": name for i, name in enumerate(self.model_config.features)}

        metadata = {
            "features": self.model_config.features,
            "feature_mapping": feature_mapping,
            "targets": self.model_config.targets,
            "problem_type": self.model_config.problem_type.name,
            "categorical_features": self.model_config.categorical_features,
            "idx": self.model_config.idx,
            "model_type": self.model_type,
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Exported metadata to {metadata_path}")

    def _export_encoders(self, output_dir: str):
        """Export encoders needed for inference."""
        import shutil

        # Copy categorical encoder
        cat_encoder_src = os.path.join(self.model_path, "vtune.categorical_encoder")
        if os.path.exists(cat_encoder_src):
            shutil.copy(cat_encoder_src, os.path.join(output_dir, "categorical_encoder.joblib"))
            logger.info("Exported categorical encoder")

        # Copy target encoder
        target_encoder_src = os.path.join(self.model_path, "vtune.target_encoder")
        if os.path.exists(target_encoder_src):
            shutil.copy(target_encoder_src, os.path.join(output_dir, "target_encoder.joblib"))
            logger.info("Exported target encoder")

    def _verify_exports(self, exported_files: List[str]):
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping verification. Install with: pip install onnxruntime")
            return

        n_features = len(self.model_config.features)
        dummy_input = np.random.randn(1, n_features).astype(np.float32)

        for filepath in exported_files:
            try:
                onnx_model = onnx.load(filepath)
                onnx.checker.check_model(onnx_model)

                session = ort.InferenceSession(filepath)
                input_name = session.get_inputs()[0].name
                _ = session.run(None, {input_name: dummy_input})

                logger.info(f"Verification passed: {os.path.basename(filepath)}")
            except Exception as e:
                logger.error(f"Verification failed for {filepath}: {e}")


def export_model(
    model_path: str,
    output_dir: Optional[str] = None,
    verify: bool = False,
):
    """Export trained VespaTune model to ONNX format."""
    exporter = VespaTuneExport(model_path=model_path)
    return exporter.export_to_onnx(output_dir=output_dir, verify=verify)
