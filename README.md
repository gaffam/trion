# Trion AI

Trion AI, yüklenen sesi analiz edip referans stile göre otomatik mixing/mastering uygulayan bir Audio AI aracıdır.

## Özellikler
- Drag & drop ses yükleme (Gradio)
- Otomatik veya manuel stil seçimi (Turkish Trap, Phonky, Hyperpop, Abbey Road, Travis Scott)
- Gerçek işleme zinciri: EQ, saturation, compression, reverb, stereo width
- Before/After dinleme
- Before/After waveform + spectrogram görselleri
- Preset export: JSON, Reaper, VST3 XML

## Kurulum
```bash
uv sync
# veya
pip install -r requirements.txt
pip install -e .
```

## CLI Kullanımı
```bash
python -m trion.ui.cli run input.wav --style auto --output processed.wav --preset json
python -m trion.ui.cli run input.wav --style "Turkish Trap" --output processed.wav --preset reaper
```

`--style` için geçerli değerler:
- `auto`
- `Turkish Trap`
- `Phonky`
- `Hyperpop`
- `Abbey Road`
- `Travis Scott`

## Gradio UI
```bash
python -m trion.ui.gradio_ui
```

Arayüzde:
- Audio yükle
- Stil seç
- Saturation/Reverb/EQ/Compressor Ratio/Stereo Width sliderlarını ayarla
- Process ile çıktıyı üret
- Processed dosyayı indir

## Proje Yapısı
`src/trion/{core,analysis,ai,processing,ui,utils,data,models,references}`
