from __future__ import annotations

import librosa
import numpy as np


def compute_fft(audio: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(audio))


def compute_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def dynamic_range(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    rms_val = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    if rms_val <= 1e-12:
        return 0.0
    return float(20 * np.log10(peak / rms_val))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def peak_amplitude(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def zero_crossing_rate(audio: np.ndarray) -> float:
    return float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))


def spectral_flatness(audio: np.ndarray) -> float:
    return float(np.mean(librosa.feature.spectral_flatness(y=audio)))


def spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    return float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))


def spectral_contrast(audio: np.ndarray, sr: int) -> float:
    return float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr)))


def spectral_rolloff(audio: np.ndarray, sr: int) -> float:
    return float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))


def mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
