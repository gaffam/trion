from __future__ import annotations

import argparse
from pathlib import Path

from trion.ai.ai_mapper import AIMapper
from trion.analysis.pro_audio_analyzer import ProAudioAnalyzer
from trion.analysis.similarity_matcher import SimilarityMatcher
from trion.processing.audio_processor import AudioProcessor
from trion.processing.preset_exporter import PresetExporter
from trion.references.default_styles import DEFAULT_STYLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("trion")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Analyze and process audio")
    run.add_argument("input", help="Input audio file path")
    run.add_argument("--style", default="auto", help="Reference style name or 'auto'")
    run.add_argument("--output", default="processed.wav", help="Processed audio output path")
    run.add_argument("--preset", default="json", choices=["json", "reaper", "vst3xml"], help="Preset export format")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd != "run":
        return

    analysis = ProAudioAnalyzer(args.input).analyze()
    matcher = SimilarityMatcher(DEFAULT_STYLES)

    if args.style.lower() == "auto":
        resolved_style, score = matcher.find_best_match(analysis)
    else:
        candidates = {k.lower(): k for k in DEFAULT_STYLES}
        if args.style.lower() not in candidates:
            available = ", ".join(DEFAULT_STYLES.keys())
            raise ValueError(f"Unknown style '{args.style}'. Available styles: {available}")
        resolved_style = candidates[args.style.lower()]
        score = matcher.find_best_match(analysis)[1]

    mapper = AIMapper()
    suggestions = mapper.suggest_processing(analysis, {resolved_style: DEFAULT_STYLES[resolved_style]})
    result = AudioProcessor().apply_suggestions(args.input, suggestions, args.output)

    preset_path = Path(args.output).with_suffix(f".{args.preset}.preset")
    PresetExporter().export(suggestions, args.preset, str(preset_path))
    print(f"Style={resolved_style} score={score:.4f} output={result.output_path} preset={preset_path}")


if __name__ == "__main__":
    main()
