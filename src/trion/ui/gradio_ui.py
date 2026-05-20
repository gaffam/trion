from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from trion.ai.ai_mapper import AIMapper
from trion.analysis.pro_audio_analyzer import ProAudioAnalyzer
from trion.analysis.similarity_matcher import SimilarityMatcher
from trion.processing.audio_processor import AudioProcessor
from trion.references.default_styles import DEFAULT_STYLES


def process_audio(
    file: str,
    style: str,
    saturation: float,
    reverb: float,
    eq_gain: float,
    compressor_ratio: float,
    stereo_width: float,
) -> tuple[str, str, str, str, str, str, str, str]:
    analysis = ProAudioAnalyzer(file).analyze()
    resolved_style = style
    if style == "auto":
        resolved_style, _ = SimilarityMatcher(DEFAULT_STYLES).find_best_match(analysis)
        resolved_style = resolved_style or "Turkish Trap"

    suggestions = AIMapper().suggest_processing(analysis, {resolved_style: DEFAULT_STYLES[resolved_style]})
    suggestions.update(
        {
            "saturation_amount": saturation,
            "reverb_mix": reverb,
            "eq_mid_gain_db": eq_gain,
            "compressor_ratio": compressor_ratio,
            "stereo_width": stereo_width,
        }
    )

    output_path = Path("processed_from_ui.wav")
    result = AudioProcessor().apply_suggestions(file, suggestions, output_path)
    return (
        file,
        str(result.output_path),
        str(result.before_waveform),
        str(result.after_waveform),
        str(result.before_spectrogram),
        str(result.after_spectrogram),
        json.dumps(suggestions, indent=2),
        str(result.output_path),
    )


with gr.Blocks(title="Trion AI") as demo:
    gr.Markdown("# Trion AI — Audio Mixing & Mastering")
    with gr.Row():
        input_audio = gr.Audio(type="filepath", label="Input Audio")
        style = gr.Dropdown(choices=["auto", *DEFAULT_STYLES.keys()], value="auto", label="Reference Style")

    with gr.Row():
        sat = gr.Slider(0.0, 0.5, 0.1, label="Saturation")
        rev = gr.Slider(0.0, 0.3, 0.05, label="Reverb Mix")
        eq = gr.Slider(-8.0, 8.0, 0.0, label="EQ Mid Gain (dB)")
        comp = gr.Slider(1.0, 10.0, 3.0, label="Compressor Ratio")
        width = gr.Slider(0.0, 0.6, 0.2, label="Stereo Width")

    process_btn = gr.Button("Process")

    with gr.Row():
        before_player = gr.Audio(label="Before")
        after_player = gr.Audio(label="After")

    with gr.Row():
        before_wave = gr.Image(label="Before Waveform")
        after_wave = gr.Image(label="After Waveform")

    with gr.Row():
        before_spec = gr.Image(label="Before Spectrogram")
        after_spec = gr.Image(label="After Spectrogram")

    preset_json = gr.Code(label="Preset (JSON)", language="json")
    download_file = gr.File(label="Download Processed Audio")

    process_btn.click(
        process_audio,
        [input_audio, style, sat, rev, eq, comp, width],
        [before_player, after_player, before_wave, after_wave, before_spec, after_spec, preset_json, download_file],
    )


if __name__ == "__main__":
    demo.launch()
