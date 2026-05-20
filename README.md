# Trion AI

Production-ready AI audio engineer workflow:
- Drag/drop upload in Gradio
- Analyze audio and auto-match style
- Apply real processing: EQ, saturation, compression, reverb, stereo imaging
- A/B listening and preset export (Reaper, VST3 XML, JSON)

## Install
```bash
uv sync
# or
pip install -r requirements.txt
pip install -e .
```

## CLI
```bash
python -m trion.ui.cli run input.wav --style "neon trap" --output processed.wav --preset reaper
```

## Gradio
```bash
python -m trion.ui.gradio_ui
```

## Structure
`src/trion/{core,analysis,ai,processing,ui,utils,data,models,references}`
