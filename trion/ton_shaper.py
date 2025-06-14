# ======================================================================
# ton_shaper.py - Simple audio post-processing utilities
# ======================================================================
"""Functions for EQ, saturation and other tone shaping operations."""

from __future__ import annotations

import numpy as np
import scipy.signal


def apply_eq(audio: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    """Apply a single band peaking EQ."""
    q = 1.0
    gain = 10 ** (gain_db / 40)
    b, a = scipy.signal.iirpeak(freq / (sr / 2), q, gain)
    return scipy.signal.lfilter(b, a, audio)


def saturate(audio: np.ndarray, amount: float = 0.5) -> np.ndarray:
    return np.tanh(audio * (1 + amount))


def compress(audio: np.ndarray, threshold_db: float = -20.0, ratio: float = 4.0) -> np.ndarray:
    linear_thresh = 10 ** (threshold_db / 20)
    gain = np.ones_like(audio)
    over = np.abs(audio) > linear_thresh
    gain[over] = (linear_thresh + (np.abs(audio[over]) - linear_thresh) / ratio) / np.abs(audio[over])
    return audio * gain


def add_reverb(audio: np.ndarray, sr: int, mix: float = 0.1) -> np.ndarray:
    ir = np.exp(-np.linspace(0, 3, int(sr * 0.5)))
    wet = np.convolve(audio, ir)[: len(audio)]
    return (1 - mix) * audio + mix * wet


def widen_stereo(audio_l: np.ndarray, audio_r: np.ndarray, width: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    mid = (audio_l + audio_r) / 2
    side = (audio_l - audio_r) / 2 * width
    left = mid + side
    right = mid - side
    return left, right
