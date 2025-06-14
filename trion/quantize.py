# ======================================================================
# quantize.py - Convert trained models to TFLite with reduced precision
# ======================================================================
"""Utility for quantizing trained models for embedded deployment."""

from __future__ import annotations

from typing import Literal

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None


def quantize(saved_model_dir: str, output_path: str, dtype: Literal["int8", "float16"] = "int8") -> str:
    if tf is None:
        raise RuntimeError("TensorFlow is required for quantization")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if dtype == "int8":
        converter.target_spec.supported_types = [tf.int8]
    elif dtype == "float16":
        converter.target_spec.supported_types = [tf.float16]
    else:
        raise ValueError("dtype must be 'int8' or 'float16'")

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path
