# ======================================================================
# model_arch.py - Tiny MLP architecture for Tri\u014dn AI
# ======================================================================
"""Lightweight model definition used for training and inference."""

from __future__ import annotations

from typing import Optional

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None


def build_model(input_dim: int, output_dim: int) -> "tf.keras.Model":
    """Build a small MLP with around 1000 parameters.

    Parameters
    ----------
    input_dim: int
        Number of input features.
    output_dim: int
        Number of output parameters.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required to build the model")

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(output_dim),
    ])
    return model
