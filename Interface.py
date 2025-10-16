# streamlit_app.py
import streamlit as st
import nmrglue as ng
import plotly.graph_objects as go
import tempfile
import os

st.title("Visualizador de Espectro ¹H-RMN (Bruker)")

# --- Upload múltiplo de arquivos ---
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
            # --- Leitura dos dados com nmrglue ---
            dic, data = ng.bruker.read_pdata(tmpdirname)
            udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
            uc = ng.fileiobase.uc_from_udic(udic, dim=0)
            ppm = uc.ppm_scale()

            # --- Plot interativo com Plotly ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ppm, y=data, mode='lines', name='Espectro', line=dict(color='black')))
            
            fig.update_layout(
                title="Espectro ¹H-RMN",
                xaxis_title="Deslocamento químico (ppm)",
                yaxis_title="Intensidade (u.a.)",
                xaxis=dict(autorange='reversed'),  # eixo ppm decrescente
                template='plotly_white',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao ler os arquivos: {e}")
