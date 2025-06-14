import os
import tempfile

import streamlit as st

from .pro_reference_manager import ProReferenceManager
from .ai_mapper import AIMapper
from .preset_exporter import PresetExporter
from .style_simulator import StyleSimulator


def main() -> None:
    st.title("Triōn AI")

    uploaded_file = st.file_uploader("Ses dosyası yükleyin", type=["wav", "flac", "mp3", "ogg"])

    ref_manager = ProReferenceManager()
    ref_names = list(ref_manager.references.keys())
    reference = st.selectbox("Referans", ["Otomatik"] + ref_names)

    model_path = st.text_input("Model path", "model.tflite")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        mapper = AIMapper(model_path=model_path if os.path.exists(model_path) else None)
        exporter = PresetExporter()
        simulator = StyleSimulator(ref_manager, mapper, exporter)

        if st.button("Analiz et"):
            suggestions = simulator.process(
                tmp_path,
                None if reference == "Otomatik" else reference,
            )
            st.subheader("Önerilen Ayarlar")
            st.json(suggestions)

            fmt = st.selectbox("Export format", ["json", "xml", "reaper", "vstpreset"])
            if st.button("Preset indir"):
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
                exporter.export(suggestions, fmt, out_file.name)
                with open(out_file.name, "rb") as f:
                    st.download_button("Download", f, file_name=os.path.basename(out_file.name))


if __name__ == "__main__":
    main()
