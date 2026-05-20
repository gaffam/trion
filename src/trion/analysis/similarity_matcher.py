from __future__ import annotations

from typing import Any
import numpy as np


class SimilarityMatcher:
    def __init__(self, references: dict[str, dict[str, Any]]) -> None:
        self.references = references

    def _vectorize(self, analysis: dict[str, Any]) -> np.ndarray:
        vec: list[float] = []
        for k in sorted(analysis.keys()):
            v = analysis[k]
            if isinstance(v, (int, float)):
                vec.append(float(v))
        return np.array(vec, dtype=np.float32)

    def _hybrid_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 1e9
        a, b = a[:n], b[:n]
        cos = 1.0 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
        euc = float(np.linalg.norm(a - b))
        return 0.6 * cos + 0.4 * euc

    def find_best_match(self, target_analysis: dict[str, Any]) -> tuple[str | None, float]:
        t = self._vectorize(target_analysis)
        best_name, best_score = None, float("inf")
        for name, ref in self.references.items():
            score = self._hybrid_distance(t, self._vectorize(ref.get("analysis", {})))
            if score < best_score:
                best_name, best_score = name, score
        return best_name, best_score
