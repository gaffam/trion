# =============================================================================
# similarity_matcher.py - Find the closest reference by audio feature distance
# =============================================================================

from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np

from .pro_reference_manager import ProReferenceManager


class SimilarityMatcher:
    """Compare analysis dictionaries to find the most similar reference."""

    def __init__(self, reference_manager: ProReferenceManager) -> None:
        self.reference_manager = reference_manager

    @staticmethod
    def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Return Euclidean distance between two analysis dicts."""
        dist = 0.0
        # Numeric keys
        for key, val in a.items():
            other = b.get(key)
            if isinstance(val, (int, float)) and isinstance(other, (int, float)):
                diff = float(val) - float(other)
                dist += diff * diff
            elif isinstance(val, list) and isinstance(other, list):
                arr_a = np.array(val, dtype=float)
                arr_b = np.array(other, dtype=float)
                if arr_a.shape == arr_b.shape:
                    diff = arr_a - arr_b
                    dist += float(np.dot(diff, diff))
        return float(np.sqrt(dist))

    def find_best_match(self, target_analysis: Dict[str, Any]) -> Tuple[str | None, float]:
        """Return the reference name with the smallest distance and the distance."""
        best_name: str | None = None
        best_dist = float("inf")
        for name, ref in self.reference_manager.references.items():
            ref_analysis = ref.get("analysis", {})
            d = self._distance(target_analysis, ref_analysis)
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name, best_dist

    def ranked_matches(self, target_analysis: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Return all references sorted by distance to the target."""
        scores = []
        for name, ref in self.reference_manager.references.items():
            ref_analysis = ref.get("analysis", {})
            d = self._distance(target_analysis, ref_analysis)
            scores.append((name, d))
        scores.sort(key=lambda x: x[1])
        return scores
