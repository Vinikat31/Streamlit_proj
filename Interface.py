# streamlit_app.py
import streamlit as st
import nmrglue as ng
import matplotlib.pyplot as plt
import tempfile
import os

st.title("Visualizador de Espectro ¹H-RMN (Bruker)")

# Upload múltiplo de arquivos
uploaded_files = st.file_uploader(
    "Selecione todos os arquivos da pasta 'pdata/1' do Bruker",
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Salva todos os arquivos temporariamente
        for file in uploaded_files:
            file_path = os.path.join(tmpdirname, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        try:
            # Lê os dados com nmrglue a partir da pasta temporária
            dic, data = ng.bruker.read_pdata(tmpdirname)
            udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
            uc = ng.fileiobase.uc_from_udic(udic, dim=0)
            ppm = uc.ppm_scale()

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ppm, data, color='black', lw=1)
            ax.invert_xaxis()
            ax.set_xlabel("Deslocamento químico (ppm)")
            ax.set_ylabel("Intensidade (u.a.)")
            ax.set_title("Espectro ¹H-RMN")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro ao ler os arquivos: {e}")
