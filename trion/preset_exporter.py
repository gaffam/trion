# =============================================================================
# preset_exporter.py - Export preset suggestions in various formats
# =============================================================================

import json
import xml.etree.ElementTree as ET
from typing import Dict


class PresetExporter:
    """Exports preset suggestions in multiple formats."""

    def export(self, settings: Dict[str, float], fmt: str, path: str) -> str:
        fmt = fmt.lower()
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4)
        elif fmt == "xml":
            root = ET.Element("preset")
            for k, v in settings.items():
                el = ET.SubElement(root, k)
                el.text = str(v)
            tree = ET.ElementTree(root)
            tree.write(path, encoding="utf-8")
        elif fmt == "reaper":
            with open(path, "w", encoding="utf-8") as f:
                for k, v in settings.items():
                    f.write(f"{k}={v}\n")
        elif fmt == "vstpreset":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings, f)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        return path
