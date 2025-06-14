"""Triōn AI modules."""

from .pro_audio_analyzer import ProAudioAnalyzer
from .pro_reference_manager import ProReferenceManager
from .style_mapper import StyleMapper
from .ai_mapper import AIMapper
from . import ai_training
from .style_simulator import StyleSimulator
from .preset_exporter import PresetExporter
from .similarity_matcher import SimilarityMatcher
from .advanced_features import (
    extract_envelope,
    extract_spectral_centroid,
    extract_spectral_flatness,
    extract_spectral_bandwidth,
    extract_spectral_contrast,
    extract_spectral_rolloff,
    extract_harmonic,
    extract_percussive,
    extract_pitch_contour,
    extract_spectrogram,
    extract_mel_spectrogram,
    extract_adsr,
    analyze_audio,
)
from .model_arch import build_model
from .train import train
from .quantize import quantize
from .ton_shaper import apply_eq, saturate, widen_stereo, compress, add_reverb
from .preset_mapper import map_metadata
from .dsp_tools import (
    compute_fft,
    compute_mfcc,
    dynamic_range,
    rms as compute_rms,
    peak_amplitude,
    zero_crossing_rate,
    spectral_centroid,
    spectral_flatness,
    spectral_bandwidth,
    mel_spectrogram,
)
from .inference import process_file
from .bluetooth_patch import apply_bt_correction
from .streamlit_gui import main as streamlit_gui
from .dataset_builder import analysis_to_flat_vector, build_dataset

__all__ = [
    "ProAudioAnalyzer",
    "ProReferenceManager",
    "StyleMapper",
    "AIMapper",
    "ai_training",
    "StyleSimulator",
    "PresetExporter",
    "SimilarityMatcher",
    "extract_envelope",
    "extract_spectral_centroid",
    "extract_spectral_flatness",
    "extract_spectral_bandwidth",
    "extract_spectral_contrast",
    "extract_spectral_rolloff",
    "extract_harmonic",
    "extract_percussive",
    "extract_pitch_contour",
    "extract_spectrogram",
    "extract_mel_spectrogram",
    "extract_adsr",
    "analyze_audio",
    "build_model",
    "train",
    "quantize",
    "apply_eq",
    "saturate",
    "widen_stereo",
    "compress",
    "add_reverb",
    "map_metadata",
    "compute_fft",
    "compute_mfcc",
    "dynamic_range",
    "compute_rms",
    "peak_amplitude",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_bandwidth",
    "mel_spectrogram",
    "process_file",
    "apply_bt_correction",
    "streamlit_gui",
    "analysis_to_flat_vector",
    "build_dataset",
]
