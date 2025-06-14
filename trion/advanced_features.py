"""Advanced audio feature extraction utilities for Triōn AI.

These functions build on :mod:`dsp_tools` to provide higher level
descriptors such as amplitude envelope, harmonic content and pitch
contour. They are used by the professional analyzer and the training
pipeline.
"""

from typing import Tuple

try:
    import crepe  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    crepe = None

from .dsp_tools import (
    rms,
    peak_amplitude,
    dynamic_range,
    zero_crossing_rate,
    spectral_centroid as basic_spectral_centroid,
    spectral_flatness as basic_spectral_flatness,
    spectral_bandwidth as basic_spectral_bandwidth,
    spectral_contrast as basic_spectral_contrast,
    spectral_rolloff as basic_spectral_rolloff,
    compute_mfcc,
    compute_fft,
)

import librosa
import soundfile as sf
import numpy as np
import scipy.signal


def extract_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return the amplitude envelope using a Hilbert transform."""
    analytic_signal = scipy.signal.hilbert(audio)
    return np.abs(analytic_signal)


def extract_adsr(envelope: np.ndarray, sr: int) -> Tuple[float, float, float, float]:
    """Return simple ADSR characteristics from an amplitude envelope."""
    if len(envelope) == 0:
        return 0.0, 0.0, 0.0, 0.0
    env = envelope / np.max(np.abs(envelope))
    attack_idx = np.argmax(env > 0.1)
    decay_idx = np.argmax(env[attack_idx:] < 0.8) + attack_idx if attack_idx < len(env) else attack_idx
    sustain_level = float(np.median(env[decay_idx:])) if decay_idx < len(env) else 0.0
    release_idx = len(env) - np.argmax(env[::-1] > 0.1) - 1
    return attack_idx / sr, decay_idx / sr, sustain_level, release_idx / sr


def extract_spectral_centroid(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return the spectral centroid over time."""
    return librosa.feature.spectral_centroid(y=audio, sr=sr)[0]


def extract_spectral_flatness(audio: np.ndarray) -> np.ndarray:
    """Return spectral flatness values over time."""
    return librosa.feature.spectral_flatness(y=audio)[0]


def extract_spectral_bandwidth(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return spectral bandwidth over time."""
    return librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]


def extract_spectral_contrast(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return spectral contrast over time."""
    return librosa.feature.spectral_contrast(y=audio, sr=sr)[0]


def extract_spectral_rolloff(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return spectral rolloff over time."""
    return librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]


def extract_harmonic(audio: np.ndarray) -> np.ndarray:
    """Return the harmonic component of the signal."""
    harmonic, _ = librosa.effects.hpss(audio)
    return harmonic


def extract_percussive(audio: np.ndarray) -> np.ndarray:
    """Return the percussive component of the signal."""
    _, percussive = librosa.effects.hpss(audio)
    return percussive


def extract_pitch_contour(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return pitch contour using CREPE if available, otherwise PYIN."""
    if crepe is not None:
        audio_f32 = librosa.util.normalize(audio.astype(np.float32))
        time, frequency, confidence, _ = crepe.predict(
            audio_f32, sr, viterbi=True, step_size=100
        )
        voiced = confidence > 0.5
        return frequency, voiced
    pitches, voiced_flags, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    return pitches, voiced_flags


def extract_spectrogram(audio: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    """Return magnitude spectrogram."""
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return spec


def extract_mel_spectrogram(audio: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    """Return mel-scaled spectrogram."""
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(spec)


def analyze_audio(file_path: str, mono: bool = True) -> dict:
    """Load audio and compute advanced features.

    Parameters
    ----------
    file_path: str
        Path to the audio file.
    mono: bool, optional
        If ``True`` the audio is mixed down to mono. If ``False`` each channel is
        averaged after feature extraction.
    """
    y_raw, sr = sf.read(file_path, dtype="float32")
    num_channels = y_raw.shape[1] if y_raw.ndim > 1 else 1
    if y_raw.ndim > 1:
        audio_mono = y_raw.mean(axis=1)
    else:
        audio_mono = y_raw
    if mono:
        audio = audio_mono
    else:
        audio = y_raw if y_raw.ndim == 1 else y_raw.T
    envelope = extract_envelope(audio, sr)
    adsr = extract_adsr(envelope, sr)
    spectral_centroid = extract_spectral_centroid(audio, sr)
    harmonic = extract_harmonic(audio)
    percussive = extract_percussive(audio)
    pitch_contour, voiced_flags = extract_pitch_contour(audio, sr)
    flatness = extract_spectral_flatness(audio)
    bandwidth = extract_spectral_bandwidth(audio, sr)
    contrast = basic_spectral_contrast(audio, sr)
    rolloff = basic_spectral_rolloff(audio, sr)
    spectrogram = extract_spectrogram(audio, sr)
    mel_spec = extract_mel_spectrogram(audio, sr)
    mfcc = compute_mfcc(audio_mono, sr, n_mfcc=13)
    fft = compute_fft(audio_mono)
    base_metrics = {
        "rms": rms(audio_mono),
        "peak_amplitude": peak_amplitude(audio_mono),
        "dynamic_range_db": dynamic_range(audio_mono),
        "zero_crossing_rate": zero_crossing_rate(audio_mono),
        "spectral_centroid_mean": basic_spectral_centroid(audio_mono, sr),
        "spectral_flatness_mean": basic_spectral_flatness(audio_mono),
        "spectral_bandwidth_mean": basic_spectral_bandwidth(audio_mono, sr),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
    }

    return {
        **base_metrics,
        "envelope": envelope.tolist(),
        "adsr": {
            "attack_sec": adsr[0],
            "decay_sec": adsr[1],
            "sustain_level": adsr[2],
            "release_sec": adsr[3],
        },
        "spectral_centroid": spectral_centroid.tolist(),
        "spectral_flatness": flatness.tolist(),
        "spectral_bandwidth": bandwidth.tolist(),
        "spectrogram": spectrogram.tolist(),
        "mel_spectrogram": mel_spec.tolist(),
        "harmonic": harmonic.tolist(),
        "percussive_rms": float(rms(percussive)),
        "pitch_contour": pitch_contour.tolist() if pitch_contour is not None else None,
        "voiced_flags": voiced_flags.tolist() if voiced_flags is not None else None,
        "mfcc": mfcc.mean(axis=1).tolist(),
        "fft": fft.tolist(),
        "spectral_contrast": contrast.tolist(),
        "spectral_rolloff": rolloff.tolist(),
        "num_channels": num_channels,
        "sr": sr,
    }



