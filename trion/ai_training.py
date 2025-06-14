# =============================================================================
# ai_training.py - training pipeline for the lightweight AI model
# =============================================================================

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from .model_arch import build_model
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None
    keras = None


def load_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from the given path.

    The directory is expected to contain numpy ``.npz`` files with ``features``
    and ``targets`` arrays. This function stacks them into one dataset.
    """
    features_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    for fname in os.listdir(data_path):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(data_path, fname))
            features_list.append(data["features"])  # shape (N, F)
            targets_list.append(data["targets"])   # shape (N, T)
    if not features_list:
        raise ValueError("No training data found in " + data_path)
    return np.vstack(features_list), np.vstack(targets_list)


def analysis_to_vector(analysis: Dict[str, Any]) -> np.ndarray:
    """Convert an analysis dictionary to a flat feature vector."""
    vec: List[float] = []
    keys = [
        "rms",
        "peak_amplitude",
        "dynamic_range_db",
        "zero_crossing_rate",
        "spectral_centroid_mean",
        "spectral_flatness_mean",
        "spectral_bandwidth_mean",
        "spectral_contrast_mean",
        "spectral_rolloff_mean",
    ]
    for k in keys:
        vec.append(float(analysis.get(k, 0.0)))
    adsr = analysis.get("adsr", {})
    vec.extend([
        float(adsr.get("attack_sec", 0.0)),
        float(adsr.get("decay_sec", 0.0)),
        float(adsr.get("sustain_level", 0.0)),
        float(adsr.get("release_sec", 0.0)),
    ])
    mfcc = analysis.get("mfcc")
    if isinstance(mfcc, list):
        vec.extend([float(x) for x in mfcc])
    return np.array(vec, dtype=np.float32)


def train(data_path: str, output_model_path: str) -> str:
    """Train a small neural network and export as a TFLite model."""
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    x, y = load_dataset(data_path)

    noise = np.random.normal(0, 0.001, size=x.shape)
    x_aug = np.concatenate([x, x + noise])
    y_aug = np.concatenate([y, y])

    model = build_model(x_aug.shape[1], y_aug.shape[1])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_aug, y_aug, epochs=25, batch_size=16)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    return output_model_path
