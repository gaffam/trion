# =============================================================================
# style_simulator.py - Orchestrates analysis and preset generation
# =============================================================================

from typing import Dict, Optional
from .pro_audio_analyzer import ProAudioAnalyzer
from .pro_reference_manager import ProReferenceManager
from .ai_mapper import AIMapper
from .preset_exporter import PresetExporter
from .similarity_matcher import SimilarityMatcher


class StyleSimulator:
    """Runs an analysis against a reference and exports preset suggestions."""

    def __init__(self,
                 reference_manager: ProReferenceManager,
                 mapper: AIMapper,
                 exporter: PresetExporter):
        self.reference_manager = reference_manager
        self.mapper = mapper
        self.exporter = exporter
        self.matcher = SimilarityMatcher(reference_manager)

    def process(
        self,
        audio_path: str,
        reference_name: Optional[str] = None,
        export_format: str | None = None,
        export_path: str | None = None,
    ) -> Dict[str, float]:
        analyzer = ProAudioAnalyzer(audio_path)
        analysis = analyzer.analyze()

        if reference_name is not None:
            reference = self.reference_manager.get_reference(reference_name)
            if reference is None:
                raise ValueError(f"Reference not found: {reference_name}")
            references = {reference_name: reference}
        else:
            best, _ = self.matcher.find_best_match(analysis)
            if best is None:
                raise ValueError("No references available")
            reference = self.reference_manager.get_reference(best)
            references = {best: reference}

        suggestions = self.mapper.suggest_processing(
            analysis,
            references,
        )

        if export_format and export_path:
            self.exporter.export(suggestions, export_format, export_path)

        return suggestions
