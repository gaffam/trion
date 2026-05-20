from __future__ import annotations
import argparse
from pathlib import Path

from trion.analysis.pro_audio_analyzer import ProAudioAnalyzer
from trion.analysis.similarity_matcher import SimilarityMatcher
from trion.ai.ai_mapper import AIMapper
from trion.processing.audio_processor import AudioProcessor
from trion.processing.preset_exporter import PresetExporter

DEFAULT_STYLES = {"Turkish Trap": {"analysis": {"rms": 0.16, "spectral_centroid": 2600, "dynamic_range_db": 10}},
                  "Hyperpop": {"analysis": {"rms": 0.20, "spectral_centroid": 3200, "dynamic_range_db": 8}},
                  "Abbey Road": {"analysis": {"rms": 0.12, "spectral_centroid": 1800, "dynamic_range_db": 13}}}


def main() -> None:
    p = argparse.ArgumentParser("trion")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("input")
    run.add_argument("--style", default="auto")
    run.add_argument("--output", default="processed.wav")
    run.add_argument("--preset", default="json")
    args = p.parse_args()

    if args.cmd == "run":
        analysis = ProAudioAnalyzer(args.input).analyze()
        matcher = SimilarityMatcher(DEFAULT_STYLES)
        style, _ = matcher.find_best_match(analysis) if args.style == "auto" else (args.style, 0.0)
        mapper = AIMapper()
        sug = mapper.suggest_processing(analysis, DEFAULT_STYLES)
        result = AudioProcessor().apply_suggestions(args.input, sug, args.output)
        PresetExporter().export(sug, args.preset if args.preset != "vst3" else "vst3xml", str(Path(args.output).with_suffix(f".{args.preset}.preset")))
        print(f"Style={style} output={result.output_path}")

if __name__ == "__main__":
    main()
