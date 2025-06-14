# ==============================================================================
# pro_reference_manager.py - Triōn AI için Referans Yönetimi (Pro Versiyon)
# ==============================================================================

import json
import os
from typing import Dict, Any, Union, List
import numpy as np

class ProReferenceManager:
    """
    Triōn AI'ın 'ses mühendisi' hafızasını yöneten ve referansları
    karşılaştıran profesyonel sınıf.
    """
    def __init__(self, reference_file: str = "trio_ai_references.json"):
        """
        Referans yöneticisini başlatır ve referansları yükler.

        Args:
            reference_file (str): Referansların kaydedileceği/yükleneceği JSON dosya adı.
        """
        self.reference_file = reference_file
        self.references: Dict[str, Dict[str, Any]] = {}
        self.load_references()

    def add_reference(self, name: str, analysis_dict: Dict[str, Any], metadata: Dict[str, str] = None):
        """
        Yeni bir referans analizi ekler veya mevcut olanı günceller.

        Args:
            name (str): Referansın benzersiz adı.
            analysis_dict (Dict[str, Any]): Ses analiz sonuçlarını içeren sözlük.
            metadata (Dict[str, str], optional): Ek metadata.
        """
        self.references[name] = {
            "analysis": analysis_dict,
            "metadata": metadata if metadata is not None else {}
        }
        print(f"Referans eklendi/güncellendi: {name}")

    def save_references(self) -> None:
        """Referansları JSON dosyasına kaydeder."""
        try:
            with open(self.reference_file, 'w', encoding='utf-8') as f:
                json.dump(self.references, f, indent=4, ensure_ascii=False)
            print(f"Referanslar başarıyla kaydedildi: {self.reference_file}")
        except IOError as e:
            print(f"Hata: Referanslar kaydedilemedi: {e}")

    def load_references(self) -> None:
        """Referansları JSON dosyasından yükler."""
        if os.path.exists(self.reference_file):
            try:
                with open(self.reference_file, 'r', encoding='utf-8') as f:
                    self.references = json.load(f)
                print(f"Referanslar başarıyla yüklendi: {self.reference_file} ({len(self.references)} adet)")
            except json.JSONDecodeError as e:
                print(f"Hata: Referans dosyası bozuk veya geçersiz JSON: {e}")
                self.references = {}
            except IOError as e:
                print(f"Hata: Referans dosyası okunamadı: {e}")
                self.references = {}
        else:
            print(f"Referans dosyası bulunamadı: {self.reference_file}. Yeni bir tane oluşturulacak.")
            self.references = {}

    def get_reference(self, name: str) -> Union[Dict[str, Any], None]:
        """Belirtilen ada sahip bir referansı döndürür."""
        return self.references.get(name)

    def compare(self, target_analysis: Dict[str, Union[float, List[float]]]) -> Dict[str, Dict[str, float]]:
        """
        Bir hedef analizi, kaydedilmiş tüm referanslarla karşılaştırır ve
        farklılıkları (mutlak değer) döndürür.
        """
        differences: Dict[str, Dict[str, float]] = {}

        numeric_target_analysis = {k: v for k, v in target_analysis.items() if isinstance(v, (int, float))}
        target_mfccs = target_analysis.get("mfccs", [])

        for ref_name, ref_data in self.references.items():
            ref_analysis = ref_data.get("analysis", {})
            diff: Dict[str, float] = {}

            for key, target_val in numeric_target_analysis.items():
                if key in ref_analysis and isinstance(ref_analysis[key], (int, float)):
                    diff[key] = abs(ref_analysis[key] - target_val)
                else:
                    diff[key] = float('inf')

            ref_mfccs = ref_analysis.get("mfccs", [])
            if target_mfccs and ref_mfccs and len(target_mfccs) == len(ref_mfccs):
                diff["mfccs_diff"] = float(np.linalg.norm(np.array(ref_mfccs) - np.array(target_mfccs)))
            elif target_mfccs or ref_mfccs:
                diff["mfccs_diff"] = float('inf')

            differences[ref_name] = diff
        return differences

