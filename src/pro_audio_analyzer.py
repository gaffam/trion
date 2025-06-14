# ==============================================================================
# pro_audio_analyzer.py - Triōn AI için Ses Özelliği Çıkarıcı (Pro Versiyon)
# ==============================================================================

import librosa
import numpy as np
import soundfile as sf
import os
from typing import Dict, Union, List

from trion.dsp_tools import compute_fft, compute_mfcc, dynamic_range
from trion.advanced_features import (
    extract_envelope,
    extract_spectral_centroid,
    extract_harmonic,
    extract_pitch_contour,
)

class ProAudioAnalyzer:
    """
    Triōn AI için profesyonel seviyede ses özelliklerini çıkaran sınıf.
    FLAC, WAV gibi çeşitli ses formatlarını destekler ve AI eğitimi için
    kritik metrikleri sağlar.
    """
    def __init__(self, file_path: str):
        """
        Analiz edilecek ses dosyasını başlatır.

        Args:
            file_path (str): Ses dosyasının yolu (örn. .flac, .wav).
        
        Raises:
            FileNotFoundError: Belirtilen dosya yolu bulunamazsa.
            sf.LibsndfileError: Ses dosyası okunamıyorsa (bozuk format vb.).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {file_path}")
        
        self.file_path = file_path
        self.y: np.ndarray
        self.sr: int

        try:
            # sf.read ile daha geniş format desteği ve hata yönetimi
            self.y, self.sr = sf.read(file_path, dtype='float32') # float32 ile daha iyi çözünürlük
            if len(self.y.shape) > 1:
                # Stereo ise ortalama alarak mono'ya indirge, ancak stereo bilgiyi koruma opsiyonu da eklenebilir.
                self.y = np.mean(self.y, axis=1)
        except sf.LibsndfileError as e:
            raise sf.LibsndfileError(f"Ses dosyası okunamadı veya format hatası: {file_path}. Hata: {e}")
        except Exception as e:
            raise Exception(f"Ses dosyasını yüklerken beklenmeyen bir hata oluştu: {file_path}. Hata: {e}")

    def get_duration(self) -> float:
        """Ses dosyasının süresini saniye cinsinden döndürür."""
        return len(self.y) / self.sr

    def get_rms(self) -> float:
        """Sesin Ortalama Karekök Gücünü (RMS) döndürür. Sesin genel enerji seviyesini temsil eder."""
        return float(np.sqrt(np.mean(self.y ** 2)))

    def get_peak_amplitude(self) -> float:
        """Sesin tepe genliğini döndürür. Dinamik aralık için kullanılır."""
        return float(np.max(np.abs(self.y)))

    def get_spectral_centroid(self) -> float:
        """Spektral Santroidi döndürür. Sesin 'parlaklığını' veya 'tizliğini' gösterir."""
        centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        return float(np.mean(centroid))

    def get_spectral_flatness(self) -> float:
        """Spektral Düzlük'ü döndürür. Sesin tınısının ne kadar 'gürültülü' veya 'müzikal' olduğunu gösterir."""
        flatness = librosa.feature.spectral_flatness(y=self.y)
        return float(np.mean(flatness))

    def get_zero_crossing_rate(self) -> float:
        """Sıfır Geçiş Oranını (ZCR) döndürür."""
        zcr = librosa.feature.zero_crossing_rate(y=self.y)
        return float(np.mean(zcr))

    def get_dynamic_range(self) -> float:
        """Sesin Dinamik Aralığını (DR) dB cinsinden döndürür."""
        return dynamic_range(self.y)

    def get_mfccs(self, n_mfcc: int = 13) -> List[float]:
        """Mel-Frequency Cepstral Coefficients (MFCCs) döndürür."""
        mfccs = compute_mfcc(self.y, self.sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1).tolist()

    def analyze(self) -> Dict[str, Union[float, List[float]]]:
        """Ses dosyasının tüm anahtar özelliklerini içeren bir sözlük döndürür."""
        envelope = extract_envelope(self.y, self.sr)
        spec_centroid = extract_spectral_centroid(self.y, self.sr)
        harmonic = extract_harmonic(self.y)
        pitch, voiced = extract_pitch_contour(self.y, self.sr)

        return {
            "duration_sec": self.get_duration(),
            "rms": self.get_rms(),
            "peak_amplitude": self.get_peak_amplitude(),
            "spectral_centroid": float(np.mean(spec_centroid)),
            "spectral_flatness": self.get_spectral_flatness(),
            "zero_crossing_rate": self.get_zero_crossing_rate(),
            "dynamic_range_db": self.get_dynamic_range(),
            "mfccs": self.get_mfccs(),
            "envelope": envelope.tolist(),
            "harmonic_rms": float(np.sqrt(np.mean(harmonic ** 2))),
            "pitch_contour": pitch.tolist() if pitch is not None else None,
            "voiced_flags": voiced.tolist() if voiced is not None else None,
            "fft": compute_fft(self.y).tolist(),
        }

