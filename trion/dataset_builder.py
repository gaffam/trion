# ======================================================================
# trion/dataset_builder.py - AI training dataset creation helpers
# ======================================================================
"""Utilities to build training datasets from audio files and references."""

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np

from .advanced_features import analyze_audio
from .pro_reference_manager import ProReferenceManager


def analysis_to_flat_vector(analysis: Dict[str, Any]) -> np.ndarray:
    """Convert analysis dictionary to a flat feature vector."""
    vec: List[float] = []
    scalar_keys = [
        "rms",
        "peak_amplitude",
        "dynamic_range_db",
        "zero_crossing_rate",
        "spectral_centroid_mean",
        "spectral_flatness_mean",
        "spectral_bandwidth_mean",
        "spectral_contrast_mean",
        "spectral_rolloff_mean",
        "harmonic_rms",
        "percussive_rms",
        "num_channels",
        "sr",
    ]
    for k in scalar_keys:
        vec.append(float(analysis.get(k, 0.0)))
    adsr = analysis.get("adsr", {})
    vec.extend([
        float(adsr.get("attack_sec", 0.0)),
        float(adsr.get("decay_sec", 0.0)),
        float(adsr.get("sustain_level", 0.0)),
        float(adsr.get("release_sec", 0.0)),
    ])
    mfcc = analysis.get("mfcc") or analysis.get("mfccs")
    if isinstance(mfcc, list):
        vec.extend(float(x) for x in mfcc)
    elif isinstance(mfcc, np.ndarray):
        vec.extend(mfcc.flatten().tolist())
    pitch = analysis.get("pitch_contour")
    if isinstance(pitch, list) and pitch:
        pitches = np.array([p for p in pitch if p and not np.isnan(p)])
        if pitches.size:
            vec.extend([
                float(np.mean(pitches)),
                float(np.min(pitches)),
                float(np.max(pitches)),
                float(np.std(pitches)),
            ])
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0])
    else:
        vec.extend([0.0, 0.0, 0.0, 0.0])
    mel_spec = analysis.get("mel_spectrogram_db") or analysis.get("mel_spectrogram")
    if isinstance(mel_spec, list) and mel_spec:
        arr = np.array(mel_spec)
        vec.extend([float(np.mean(arr)), float(np.std(arr))])
    else:
        vec.extend([0.0, 0.0])
    fft = analysis.get("fft_magnitude") or analysis.get("fft")
    if isinstance(fft, list) and fft:
        arr = np.array(fft)
        vec.extend([float(np.mean(arr)), float(np.std(arr))])
    else:
        vec.extend([0.0, 0.0])
    return np.array(vec, dtype=np.float32)


def generate_target_vector(
    target_params_template: Dict[str, float]
) -> np.ndarray:
    """Create a target vector from a parameter template."""
    keys = [
        "eq_mid_freq",
        "eq_mid_gain_db",
        "compressor_threshold_db",
        "saturation_amount",
        "reverb_mix",
        "stereo_width_ratio",
        "low_pass_freq",
        "high_pass_freq",
        "adsr_attack_target",
        "adsr_decay_target",
        "harmonic_boost_amount",
        "percussive_boost_amount",
    ]
    return np.array([float(target_params_template.get(k, 0.0)) for k in keys], dtype=np.float32)


def build_dataset(
    audio_files_dir: str,
    output_dir: str,
    reference_manager: ProReferenceManager,
    target_processing_templates: Dict[str, Dict[str, float]],
) -> None:
    """Generate .npz training data from audio files and templates."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for name, ref in reference_manager.references.items():
        file_path = ref.get("metadata", {}).get("file_path")
        if not file_path:
            continue
        if not os.path.isabs(file_path):
            file_path = os.path.join(audio_files_dir, file_path)
        if not os.path.exists(file_path):
            continue
        try:
            analysis = analyze_audio(file_path, mono=True)
        except Exception:
            continue
        features = analysis_to_flat_vector(analysis)
        targets = generate_target_vector(target_processing_templates.get(name, {}))
        out_path = os.path.join(output_dir, f"{name}_data.npz")
        np.savez_compressed(out_path, features=features, targets=targets)
    print("Dataset creation complete.")

__all__ = ["analysis_to_flat_vector", "build_dataset"]
