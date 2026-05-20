from __future__ import annotations
from typing import Any
import numpy as np


class AIMapper:
    def suggest_processing(self, target_analysis: dict[str, Any], references: dict[str, dict[str, Any]]) -> dict[str, float]:
        rms = float(target_analysis.get("rms", 0.0))
        centroid = float(target_analysis.get("spectral_centroid", 0.0))
        dyn = float(target_analysis.get("dynamic_range_db", 0.0))
        return {
            "eq_mid_gain_db": -2.0 if centroid > 2500 else 2.0,
            "saturation_amount": float(np.clip((0.25 - rms) * 2.0, 0.05, 0.3)),
            "compressor_threshold_db": -22.0 if dyn > 12 else -14.0,
            "compressor_ratio": 3.0,
            "reverb_mix": 0.04 if rms > 0.12 else 0.08,
            "stereo_width": 0.20,
        }
