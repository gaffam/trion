from __future__ import annotations

DEFAULT_STYLES: dict[str, dict[str, dict[str, float]]] = {
    "Turkish Trap": {"analysis": {"rms": 0.16, "spectral_centroid": 2600.0, "dynamic_range_db": 10.0}},
    "Phonky": {"analysis": {"rms": 0.18, "spectral_centroid": 2400.0, "dynamic_range_db": 9.0}},
    "Hyperpop": {"analysis": {"rms": 0.20, "spectral_centroid": 3200.0, "dynamic_range_db": 8.0}},
    "Abbey Road": {"analysis": {"rms": 0.12, "spectral_centroid": 1800.0, "dynamic_range_db": 13.0}},
    "Travis Scott": {"analysis": {"rms": 0.17, "spectral_centroid": 2200.0, "dynamic_range_db": 9.5}},
}
