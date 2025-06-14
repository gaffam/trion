# =============================================================================
# interface_cli.py - Simple command line interface for Triōn
# =============================================================================

import argparse
from .pro_reference_manager import ProReferenceManager
from .ai_mapper import AIMapper
from .preset_exporter import PresetExporter
from .style_simulator import StyleSimulator
from . import ai_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Triōn command line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Analyze audio and suggest processing")
    run_p.add_argument("audio_file", help="Path to the audio file to analyze")
    run_p.add_argument(
        "--reference",
        help="Name of the reference to compare (optional, will auto select)",
    )
    run_p.add_argument("--format", default="json", help="Export format")
    run_p.add_argument("--output", help="Output preset file path")
    run_p.add_argument(
        "--model_path",
        default="model.tflite",
        help="Path to the AI model used by the mapper",
    )

    train_p = sub.add_parser("train", help="Train the AI model")
    train_p.add_argument("--data_path", required=True, help="Path to training dataset")
    train_p.add_argument("--output_model_path", required=True, help="Where to save the trained model")

    args = parser.parse_args()

    if args.command == "run":
        ref_manager = ProReferenceManager()
        mapper = AIMapper(model_path=args.model_path)
        exporter = PresetExporter()
        simulator = StyleSimulator(ref_manager, mapper, exporter)

        suggestions = simulator.process(
            args.audio_file,
            reference_name=args.reference,
            export_format=args.format if args.output else None,
            export_path=args.output,
        )

        print("Suggested settings:")
        for k, v in suggestions.items():
            print(f"  {k}: {v}")
    elif args.command == "train":
        ai_training.train(args.data_path, args.output_model_path)


if __name__ == "__main__":
    main()
