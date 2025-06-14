# =============================================================================
# ai_mapper.py - AI powered mapping of audio differences to processing parameters
# =============================================================================

from __future__ import annotations

import os
from typing import Any, Dict, List, Union, Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None


class AIMapper:
    """Map analysis differences to processing suggestions using a lightweight AI model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.interpreter: Optional[tf.lite.Interpreter] = None
        if tf is not None and model_path and os.path.exists(model_path):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

    def _run_model(self, features: np.ndarray) -> Optional[np.ndarray]:
        if self.interpreter is None:
            return None
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]["index"], features.astype(np.float32))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(output_details[0]["index"])
        return output.squeeze()

    def suggest_processing(
        self,
        target_analysis: Dict[str, Union[float, List[float]]],
        references: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Return processing suggestions based on target analysis and references."""
        # Aggregate reference analyses for simple feature context
        if references:
            ref_vals = [r.get("analysis", {}) for r in references.values()]
            avg_ref = {
                k: float(np.mean([rv.get(k, 0.0) for rv in ref_vals]))
                for k in target_analysis
                if isinstance(target_analysis[k], (int, float))
            }
        else:
            avg_ref = {k: 0.0 for k in target_analysis if isinstance(target_analysis[k], (int, float))}

        feature_keys = sorted(avg_ref)
        features = np.array([[target_analysis.get(k, 0.0) - avg_ref.get(k, 0.0) for k in feature_keys]])
        model_output = self._run_model(features)

        if model_output is not None:
            keys = [
                "eq_mid_freq",
                "eq_mid_gain_db",
                "compressor_threshold_db",
                "saturation_amount",
                "reverb_mix",
            ]
            return {k: float(v) for k, v in zip(keys, model_output.tolist())}

        # Fallback heuristic if model not available
        suggestions: Dict[str, float] = {}
        idx = {k: i for i, k in enumerate(feature_keys)}
        centroid_diff = features[0][idx.get("spectral_centroid", 0)]
        flatness_diff = features[0][idx.get("spectral_flatness_mean", 0)] if "spectral_flatness_mean" in idx else 0.0
        rms_diff = features[0][idx.get("rms", 0)]
        dyn_diff = features[0][idx.get("dynamic_range_db", 0)] if "dynamic_range_db" in idx else 0.0
        contrast_diff = features[0][idx.get("spectral_contrast_mean", 0)] if "spectral_contrast_mean" in idx else 0.0
        rolloff_diff = features[0][idx.get("spectral_rolloff_mean", 0)] if "spectral_rolloff_mean" in idx else 0.0

        if centroid_diff > 300:
            suggestions["eq_mid_freq"] = 2000.0
            suggestions["eq_mid_gain_db"] = -3.0
        elif centroid_diff < -300:
            suggestions["eq_mid_freq"] = 800.0
            suggestions["eq_mid_gain_db"] = 3.0

        if rms_diff > 0.1:
            suggestions["compressor_threshold_db"] = -20.0
        elif rms_diff < -0.1:
            suggestions["compressor_threshold_db"] = -10.0

        if flatness_diff > 0.1:
            suggestions["saturation_amount"] = 0.2
        else:
            suggestions["saturation_amount"] = 0.1

        if contrast_diff < -2:
            suggestions["eq_high_shelf_db"] = 2.0
        elif contrast_diff > 2:
            suggestions["eq_high_shelf_db"] = -2.0

        if rolloff_diff < -500:
            suggestions["low_pass_freq"] = 8000.0
        elif rolloff_diff > 500:
            suggestions["low_pass_freq"] = 12000.0

        if dyn_diff < -3:
            suggestions["reverb_mix"] = 0.03
        elif dyn_diff > 3:
            suggestions["reverb_mix"] = 0.07
        else:
            suggestions["reverb_mix"] = 0.05
        return suggestions
