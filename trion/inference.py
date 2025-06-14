# ======================================================================
# inference.py - Run trained model on audio chunks
# ======================================================================
"""Pipeline for applying the trained model to audio."""

from __future__ import annotations

from typing import List, Dict

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None

from .pro_audio_analyzer import ProAudioAnalyzer
from .ton_shaper import apply_eq, saturate


def process_file(audio_path: str, model_path: str) -> Dict[str, float]:
    if tf is None:
        raise RuntimeError("TensorFlow is required for inference")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    analyzer = ProAudioAnalyzer(audio_path)
    analysis = analyzer.analyze()

    features = np.array([[analysis.get(k, 0.0) for k in sorted(analysis) if isinstance(analysis[k], (int, float))]])
    interpreter.set_tensor(input_details[0]["index"], features.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"]).squeeze().tolist()

    # simple post processing using ton_shaper
    processed = {
        "eq_mid_freq": output[0],
        "eq_mid_gain_db": output[1],
        "compressor_threshold_db": output[2],
        "saturation_amount": output[3],
        "reverb_mix": output[4],
    }
    return processed
