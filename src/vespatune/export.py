import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import onnx

from .enums import ProblemType
from .logger import logger
from .models import get_model


@dataclass
class VespaTuneExport:
    model_path: str

    def __post_init__(self):
        self.model_config = joblib.load(os.path.join(self.model_path, "vtune.config"))
        self.raw_model = joblib.load(os.path.join(self.model_path, "vtune_model.final"))
        self.model_type = getattr(self.model_config, "model_type", "xgboost")
        self.n_features = len(self.model_config.features)

    def _wrap_model(self, raw_model):
        """Wrap a raw model in the appropriate model class for ONNX export."""
        model = get_model(
            model_name=self.model_type,
            problem_type=self.model_config.problem_type.name,
            random_state=42,
        )
        model.model = raw_model
        return model

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
            # Multiple models, one per target
            for idx, target in enumerate(self.model_config.targets):
                model = self._wrap_model(self.raw_model[idx])
                onnx_model = model.to_onnx(self.n_features)
                output_path = os.path.join(output_dir, f"model_{target}.onnx")
                onnx.save_model(onnx_model, output_path)
                exported_files.append(output_path)
                logger.info(f"Exported model for target '{target}' to {output_path}")
        else:
            # Single model
            model = self._wrap_model(self.raw_model)
            onnx_model = model.to_onnx(self.n_features)
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
        """Export metadata for inference."""
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
        cat_encoder_src = os.path.join(self.model_path, "vtune.categorical_encoder")
        if os.path.exists(cat_encoder_src):
            shutil.copy(cat_encoder_src, os.path.join(output_dir, "categorical_encoder.joblib"))
            logger.info("Exported categorical encoder")

        target_encoder_src = os.path.join(self.model_path, "vtune.target_encoder")
        if os.path.exists(target_encoder_src):
            shutil.copy(target_encoder_src, os.path.join(output_dir, "target_encoder.joblib"))
            logger.info("Exported target encoder")

    def _verify_exports(self, exported_files: List[str]):
        """Verify exported ONNX models."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed. Skipping verification.")
            return

        dummy_input = np.random.randn(1, self.n_features).astype(np.float32)

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
