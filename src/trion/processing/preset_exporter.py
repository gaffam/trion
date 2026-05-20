from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from pathlib import Path


class PresetExporter:
    def export(self, settings: dict[str, float], fmt: str, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt == "json":
            p.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        elif fmt == "reaper":
            p.write_text("\n".join([f"{k}={v}" for k, v in settings.items()]), encoding="utf-8")
        elif fmt in {"vst3xml", "xml"}:
            root = ET.Element("vst3_preset")
            for k, v in settings.items():
                ET.SubElement(root, "param", name=k, value=str(v))
            ET.ElementTree(root).write(p, encoding="utf-8")
        else:
            raise ValueError(fmt)
        return str(p)
