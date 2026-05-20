from __future__ import annotations
import gradio as gr
from trion.ui.cli import DEFAULT_STYLES
from trion.analysis.pro_audio_analyzer import ProAudioAnalyzer
from trion.ai.ai_mapper import AIMapper
from trion.processing.audio_processor import AudioProcessor


def process_audio(file: str, style: str, saturation: float, reverb: float):
    analysis = ProAudioAnalyzer(file).analyze()
    sug = AIMapper().suggest_processing(analysis, DEFAULT_STYLES)
    sug["saturation_amount"] = saturation
    sug["reverb_mix"] = reverb
    out = "processed_from_ui.wav"
    AudioProcessor().apply_suggestions(file, sug, out)
    return file, out, str(sug)

with gr.Blocks(title="Trion AI") as demo:
    gr.Markdown("# Trion AI - Audio Mixing/Mastering")
    inp = gr.Audio(type="filepath", label="Input")
    style = gr.Dropdown(choices=["auto", "Turkish Trap", "Phonky", "Hyperpop", "Abbey Road", "Travis Scott"], value="auto")
    sat = gr.Slider(0.0, 0.5, 0.1, label="Saturation")
    rev = gr.Slider(0.0, 0.3, 0.05, label="Reverb Mix")
    btn = gr.Button("Process")
    before = gr.Audio(label="Before")
    after = gr.Audio(label="After")
    txt = gr.Textbox(label="Preset")
    btn.click(process_audio, [inp, style, sat, rev], [before, after, txt])

if __name__ == "__main__":
    demo.launch()
