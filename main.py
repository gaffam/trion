# ==============================================================================
# main.py - Triōn AI'ın Ana Giriş Noktası
# ==============================================================================

import argparse
import json
import sys
import os

from trion.pro_audio_analyzer import ProAudioAnalyzer
from trion.pro_reference_manager import ProReferenceManager
from trion.ai_mapper import AIMapper
from trion.preset_exporter import PresetExporter
from trion.style_simulator import StyleSimulator
from trion.similarity_matcher import SimilarityMatcher
from trion.advanced_features import (
    analyze_audio,
    extract_envelope,
    extract_harmonic,
    extract_pitch_contour,
    extract_spectral_centroid,
)
from trion import ai_training


def main() -> None:
    """Triōn AI command line interface."""
    parser = argparse.ArgumentParser(
        description="Triōn AI Komut Satırı Arayüzü: Müzik için Akıllı Ses Mühendisi"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Ses dosyasını analiz et ve stil önerileri üret"
    )
    run_parser.add_argument("audio_file", help="Analiz edilecek ses dosyası")
    run_parser.add_argument(
        "--reference",
        help=(
            "Karşılaştırılacak referansın adı. Boş bırakılırsa en yakın referans"
            " otomatik seçilir."
        ),
    )
    run_parser.add_argument(
        "--output_format",
        default="json",
        choices=["json", "xml", "reaper", "vstpreset"],
        help="Çıktı önayar dosyasının formatı",
    )
    run_parser.add_argument(
        "--output_path", help="Çıktı önayar dosyasının kaydedileceği yol"
    )
    run_parser.add_argument(
        "--model_path",
        default="model.tflite",
        help="AI Mapper için kullanılacak TFLite modeli",
    )

    train_parser = subparsers.add_parser("train", help="Triōn AI modelini eğit")
    train_parser.add_argument(
        "--data_path", required=True, help="Eğitim veri setinin bulunduğu dizin"
    )
    train_parser.add_argument(
        "--output_model_path",
        default="model.tflite",
        help="Eğitilmiş modelin kaydedileceği yol",
    )

    add_ref_parser = subparsers.add_parser(
        "add-ref", help="Yeni bir referans ses kaydı ekle"
    )
    add_ref_parser.add_argument("name", help="Referansın adı")
    add_ref_parser.add_argument("audio_file", help="Ses dosyasının yolu")
    add_ref_parser.add_argument(
        "--metadata",
        type=json.loads,
        help='JSON formatında metadata (örneğin: {"studio": "Abbey Road"})',
    )

    args = parser.parse_args()

    if args.command == "run":
        mapper = AIMapper(model_path=args.model_path)
        ref_manager = ProReferenceManager()
        exporter = PresetExporter()
        simulator = StyleSimulator(ref_manager, mapper, exporter)

        try:
            suggestions = simulator.process(
                audio_path=args.audio_file,
                reference_name=args.reference,
                export_format=args.output_format if args.output_path else None,
                export_path=args.output_path,
            )
            print("\nÖnerilen Ayarlar:")
            for k, v in suggestions.items():
                print(f"  {k}: {v:.4f}")
            if args.output_path:
                print(f"\nAyarlar kaydedildi: {args.output_path}")
        except Exception as e:
            print(f"Hata: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "train":
        try:
            trained_model_path = ai_training.train(
                args.data_path, args.output_model_path
            )
            print(
                f"AI modeli başarıyla eğitildi ve kaydedildi: {trained_model_path}"
            )
        except Exception as e:
            print(f"Hata: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "add-ref":
        ref_manager = ProReferenceManager()
        try:
            print(
                f"'{args.name}' referansı için ses analizi yapılıyor (gelişmiş özellikler dahil)..."
            )
            analysis_results = analyze_audio(args.audio_file, mono=False)
            ref_manager.add_reference(args.name, analysis_results, args.metadata)
            ref_manager.save_references()
            print(f"Referans '{args.name}' başarıyla eklendi.")
        except Exception as e:
            print(f"Hata: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
