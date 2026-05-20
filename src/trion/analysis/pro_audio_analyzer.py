from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import librosa
import numpy as np


@dataclass
class ProAudioAnalyzer:
    file_path: str

    def analyze(self) -> dict[str, Any]:
        y, sr = librosa.load(self.file_path, sr=None, mono=True)
        return {
            "duration_sec": float(len(y) / sr),
            "rms": float(np.sqrt(np.mean(y**2))),
            "peak_amplitude": float(np.max(np.abs(y))),
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=y))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
            "dynamic_range_db": float(20 * np.log10((np.max(np.abs(y)) + 1e-8) / (np.sqrt(np.mean(y**2)) + 1e-8))),
        }
