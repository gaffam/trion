# ======================================================================
# dsp_tools.py - Basic DSP helper functions
# ======================================================================
"""Utility functions for common digital signal processing tasks.

These helpers provide basic audio metrics like RMS, peak amplitude,
zero crossing rate and spectral statistics. They are lightweight so
other modules can reuse them without pulling in large dependencies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import librosa
import scipy.signal


def compute_fft(audio: np.ndarray) -> np.ndarray:
    """Return magnitude spectrum of the audio using STFT."""
    spec = librosa.stft(audio)
    return np.abs(spec)


def compute_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def dynamic_range(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    rms_val = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if rms_val <= 1e-9:
        return 0.0
    return float(20 * np.log10(peak / rms_val))


def rms(audio: np.ndarray) -> float:
    """Return root mean square of the signal."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def peak_amplitude(audio: np.ndarray) -> float:
    """Return peak absolute amplitude of the signal."""
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Average zero crossing rate of the signal."""
    return float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Mean spectral centroid."""
    return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))


def spectral_flatness(audio: np.ndarray) -> float:
    """Mean spectral flatness."""
    return float(np.mean(librosa.feature.spectral_flatness(y=audio)))


def spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Mean spectral bandwidth."""
    return float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))


def spectral_contrast(audio: np.ndarray, sr: int) -> float:
    """Mean spectral contrast."""
    return float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr)))


def spectral_rolloff(audio: np.ndarray, sr: int) -> float:
    """Mean spectral rolloff frequency."""
    return float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))


def mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """Mel-scaled spectrogram."""
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)


