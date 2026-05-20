from __future__ import annotations

import librosa.display

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from loguru import logger


@dataclass
class ProcessResult:
    output_path: Path
    before_waveform: Path
    after_waveform: Path
    before_spectrogram: Path
    after_spectrogram: Path


class AudioProcessor:
    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr

    def apply_suggestions(self, input_path: str | Path, suggestions: dict[str, float], output_path: str | Path) -> ProcessResult:
        audio, sr = librosa.load(str(input_path), sr=self.sr, mono=False)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, 0)
        processed = audio.copy()

        gain = float(suggestions.get("eq_mid_gain_db", 0.0))
        processed *= 10 ** (gain / 20.0)
        sat = float(suggestions.get("saturation_amount", 0.1))
        processed = np.tanh(processed * (1.0 + sat * 4.0))

        threshold = float(suggestions.get("compressor_threshold_db", -18.0))
        ratio = float(suggestions.get("compressor_ratio", 3.0))
        thresh_linear = 10 ** (threshold / 20.0)
        above = np.abs(processed) > thresh_linear
        processed[above] = np.sign(processed[above]) * (thresh_linear + (np.abs(processed[above]) - thresh_linear) / ratio)

        reverb_mix = float(suggestions.get("reverb_mix", 0.05))
        impulse = np.zeros(int(sr * 0.2), dtype=np.float32)
        impulse[0] = 1.0
        impulse[:: max(1, len(impulse)//20)] += np.linspace(0.4, 0.05, len(impulse[:: max(1, len(impulse)//20)]))
        wet = np.vstack([fftconvolve(ch, impulse, mode="full")[: ch.shape[0]] for ch in processed])
        processed = (1 - reverb_mix) * processed + reverb_mix * wet

        width = float(suggestions.get("stereo_width", 0.15))
        if processed.shape[0] >= 2:
            mid = (processed[0] + processed[1]) * 0.5
            side = (processed[0] - processed[1]) * (0.5 + width)
            processed[0] = mid + side
            processed[1] = mid - side

        processed = np.clip(processed, -1.0, 1.0)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, processed.T, sr)

        assets = out.parent / f"{out.stem}_assets"
        assets.mkdir(parents=True, exist_ok=True)
        before_wave = assets / "before_waveform.png"
        after_wave = assets / "after_waveform.png"
        before_spec = assets / "before_spectrogram.png"
        after_spec = assets / "after_spectrogram.png"

        self._save_waveform(audio, sr, before_wave)
        self._save_waveform(processed, sr, after_wave)
        self._save_spectrogram(audio, sr, before_spec)
        self._save_spectrogram(processed, sr, after_spec)
        logger.info("Processed audio saved to {}", out)
        return ProcessResult(out, before_wave, after_wave, before_spec, after_spec)

    def _save_waveform(self, audio: np.ndarray, sr: int, path: Path) -> None:
        plt.figure(figsize=(12, 3))
        librosa.display.waveshow(audio[0], sr=sr)
        plt.tight_layout(); plt.savefig(path); plt.close()

    def _save_spectrogram(self, audio: np.ndarray, sr: int, path: Path) -> None:
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio[0])), ref=np.max)
        plt.figure(figsize=(12, 3))
        librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout(); plt.savefig(path); plt.close()
