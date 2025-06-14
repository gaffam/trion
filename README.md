# Trion

This repository contains the early modules for Triōn AI. It currently includes:

- `pro_audio_analyzer.py`: advanced audio feature extraction using `librosa` and `soundfile`.
- `pro_reference_manager.py`: a reference manager for storing and comparing audio analysis data.
- `style_mapper.py`: simple mapping from differences to EQ/compression.
- `ai_mapper.py`: AI powered suggestion engine that can run as a TFLite model.
- `style_simulator.py`: orchestrates analysis and generates preset suggestions.
- `ai_training.py`: tools to train the lightweight model for `ai_mapper`.
- `interface_cli.py`: simple command line helper.
- `streamlit_gui.py`: optional Streamlit UI to run the analysis from a browser.
- `main.py`: unified CLI entry point for analysis, training and reference
  management.
- `preset_exporter.py`: writes suggestions in `.json`, `.xml`, `.reaper`, or `.vstpreset` formats.
- `similarity_matcher.py`: finds the closest reference based on analysis features.
- `advanced_features.py`: higher level extraction utilities for envelope,
  ADSR estimation, harmonic and percussive separation, pitch contour (CREPE
  fallback to PYIN) along with spectral statistics (contrast, rolloff) and
  mel/FFT spectrogram generation.
- `dsp_tools.py`: simple DSP helpers for STFT-based FFT, RMS, peak level,
  zero crossing rate and core spectral metrics including contrast and rolloff.
- `model_arch.py`: tiny neural network used for training and inference.
- `train.py`: utility to train the lightweight model with simple augmentation.
- `quantize.py`: converts a trained model to TFLite int8/float16.
 - `ton_shaper.py`, `preset_mapper.py`: processing helpers for EQ, saturation,
   compression and reverb along with metadata mapping.
 - `inference.py`: run a TFLite model on an audio file.
 - `bluetooth_patch.py`: compensates for Bluetooth codec losses and shapes the spectrum.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

These modules rely on `librosa`, `numpy`, `soundfile`, `scipy` and `tensorflow` for the optional AI features. Pitch detection uses `crepe` if installed.

## CLI usage

Analyze audio and export preset suggestions:

```bash
python main.py run input.wav --reference my_ref --output_path preset.json
```

Train the AI model from a dataset of ``.npz`` files:

```bash
python main.py train --data_path ./dataset --output_model_path model.tflite
```

Add a new reference recording:

```bash
python main.py add-ref my_ref path/to/audio.wav --metadata '{"studio": "Abbey Road"}'
```

Launch the Streamlit GUI:

```bash
streamlit run -m trion.streamlit_gui
```
