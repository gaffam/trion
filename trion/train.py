# ======================================================================
# train.py - Training utilities for Tri\u014dn AI model
# ======================================================================
"""Training script for the lightweight model with simple augmentation."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - tensorflow may not be installed
    tf = None

from .model_arch import build_model


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from .npz files in ``path``."""
    feats = []
    labels = []
    for fname in os.listdir(path):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(path, fname))
            feats.append(data["features"])
            labels.append(data["labels"])
    return np.vstack(feats), np.vstack(labels)


def augment(features: np.ndarray) -> np.ndarray:
    noise = np.random.normal(scale=0.01, size=features.shape)
    return features + noise


def train(data_path: str, out_path: str, epochs: int = 10) -> str:
    if tf is None:
        raise RuntimeError("TensorFlow is required for training")

    x, y = load_dataset(data_path)
    x_aug = augment(x)
    x_train = np.concatenate([x, x_aug], axis=0)
    y_train = np.concatenate([y, y], axis=0)

    model = build_model(x.shape[1], y.shape[1])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, batch_size=32, epochs=epochs)
    model.save(out_path)
    return out_path
