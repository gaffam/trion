# ======================================================================
# bluetooth_patch.py - Simple correction for Bluetooth codec artifacts
# ======================================================================
"""Applies EQ adjustments to compensate for codec loss."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def apply_bt_correction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Boost high frequencies to compensate for Bluetooth compression."""
    b, a = butter(2, 4000 / (sr / 2), btype="high")
    high = filtfilt(b, a, audio)
    low_b, low_a = butter(2, 200 / (sr / 2), btype="low")
    low = filtfilt(low_b, low_a, audio)
    corrected = audio + 0.5 * high - 0.1 * low
    return corrected
