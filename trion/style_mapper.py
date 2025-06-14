# =============================================================================
# style_mapper.py - Suggest processing adjustments based on analysis differences
# =============================================================================

from typing import Dict

class StyleMapper:
    """Maps analysis differences to simple processing suggestions."""

    def __init__(self,
                 centroid_thresh: float = 500.0,
                 rms_thresh: float = 0.1,
                 dynamic_range_thresh: float = 3.0):
        self.centroid_thresh = centroid_thresh
        self.rms_thresh = rms_thresh
        self.dynamic_range_thresh = dynamic_range_thresh

    def map_differences(self,
                        target: Dict[str, float],
                        reference: Dict[str, float]) -> Dict[str, float]:
        """Return a dict of processing suggestions."""
        suggestions: Dict[str, float] = {}

        centroid_diff = target.get("spectral_centroid", 0) - reference.get("spectral_centroid", 0)
        if centroid_diff > self.centroid_thresh:
            suggestions["eq_high_shelf_db"] = -3.0
        elif centroid_diff < -self.centroid_thresh:
            suggestions["eq_high_shelf_db"] = 3.0

        rms_diff = target.get("rms", 0) - reference.get("rms", 0)
        if rms_diff > self.rms_thresh:
            suggestions["compressor_ratio"] = 4.0
        elif rms_diff < -self.rms_thresh:
            suggestions["compressor_ratio"] = 2.0

        dr_diff = target.get("dynamic_range_db", 0) - reference.get("dynamic_range_db", 0)
        if dr_diff < -self.dynamic_range_thresh:
            suggestions["expander_ratio"] = 1.5
        elif dr_diff > self.dynamic_range_thresh:
            suggestions["limiter_threshold_db"] = -1.0

        return suggestions
