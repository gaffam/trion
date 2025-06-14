# ======================================================================
# preset_mapper.py - Map metadata to preset templates
# ======================================================================
"""Provides preset defaults based on reference metadata."""

from __future__ import annotations

from typing import Dict

_PRESETS = {
    "abbey_road": {"eq_mid_freq": 1000.0, "reverb_mix": 0.1},
    "sunset_sound": {"eq_mid_freq": 1500.0, "reverb_mix": 0.05},
}


def map_metadata(metadata: Dict[str, str]) -> Dict[str, float]:
    studio = metadata.get("studio", "").lower()
    return _PRESETS.get(studio, {})
