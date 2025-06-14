# ======================================================================
# tests/test_dsp_tools.py - tests for trion.dsp_tools
# ======================================================================
"""Pytest tests for the dsp_tools module."""

import numpy as np
import pytest

from trion import dsp_tools

SR = 44100
DURATION = 1.0
FREQ = 440.0
T = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
SINE_WAVE = 0.5 * np.sin(2 * np.pi * FREQ * T)
NOISY_SINE_WAVE = SINE_WAVE + np.random.normal(0, 0.1, SINE_WAVE.shape)
SILENT_WAVE = np.zeros_like(SINE_WAVE)


def test_rms():
    assert pytest.approx(dsp_tools.rms(SINE_WAVE), rel=1e-3) == 0.5 / np.sqrt(2)
    assert dsp_tools.rms(SILENT_WAVE) == 0.0
    assert dsp_tools.rms(np.array([])) == 0.0


def test_peak_amplitude():
    assert pytest.approx(dsp_tools.peak_amplitude(SINE_WAVE)) == 0.5
    assert dsp_tools.peak_amplitude(SILENT_WAVE) == 0.0
    assert dsp_tools.peak_amplitude(np.array([])) == 0.0


def test_dynamic_range():
    expected = 20 * np.log10(np.sqrt(2))
    assert pytest.approx(dsp_tools.dynamic_range(SINE_WAVE), abs=0.1) == expected
    assert dsp_tools.dynamic_range(SILENT_WAVE) == 0.0


def test_zero_crossing_rate():
    assert dsp_tools.zero_crossing_rate(SINE_WAVE) > 0.01
    assert dsp_tools.zero_crossing_rate(SILENT_WAVE) < 0.01


def test_compute_fft():
    fft_result = dsp_tools.compute_fft(SINE_WAVE)
    assert fft_result.ndim == 2
    assert np.all(fft_result >= 0)
    assert fft_result.shape[0] > 0 and fft_result.shape[1] > 0


def test_compute_mfcc():
    mfcc_result = dsp_tools.compute_mfcc(SINE_WAVE, SR)
    assert mfcc_result.ndim == 2
    assert mfcc_result.shape[0] == 13
    assert mfcc_result.shape[1] > 0


def test_spectral_centroid():
    centroid = dsp_tools.spectral_centroid(SINE_WAVE, SR)
    assert pytest.approx(centroid, rel=0.1) == FREQ
    assert centroid > 0.0


def test_spectral_flatness():
    flatness = dsp_tools.spectral_flatness(SINE_WAVE)
    assert flatness < 0.1
    assert flatness >= 0.0


def test_spectral_bandwidth():
    bandwidth = dsp_tools.spectral_bandwidth(SINE_WAVE, SR)
    assert bandwidth > 0.0
    assert bandwidth < 1000.0


def test_spectral_contrast():
    contrast = dsp_tools.spectral_contrast(SINE_WAVE, SR)
    noisy_contrast = dsp_tools.spectral_contrast(NOISY_SINE_WAVE, SR)
    assert contrast >= 0.0
    assert noisy_contrast < contrast


def test_spectral_rolloff():
    rolloff = dsp_tools.spectral_rolloff(SINE_WAVE, SR)
    assert rolloff > FREQ
    assert rolloff < SR / 2


def test_mel_spectrogram():
    mel_spec_result = dsp_tools.mel_spectrogram(SINE_WAVE, SR, n_mels=64)
    assert mel_spec_result.ndim == 2
    assert mel_spec_result.shape[0] == 64
    assert np.all(mel_spec_result >= 0.0)
