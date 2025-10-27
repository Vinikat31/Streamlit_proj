import streamlit as st
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import os
import tempfile
import zipfile
import shutil

st.set_page_config(page_title="Visualizador de RMN", layout="wide")
st.title("🧪 Visualizador de Espectros de RMN")

# --- Upload do ZIP da pasta raiz ---
uploaded_zip = st.sidebar.file_uploader(
    "📤 Envie a pasta raiz dos espectros (compactada em ZIP)",
    type="zip"
)

def extract_number(s):
    """Extrai números do nome da pasta para ordenar corretamente.
       Se não houver número, retorna +inf para ficar no final."""
    nums = ''.join(filter(str.isdigit, s))
    return int(nums) if nums else float('inf')

if uploaded_zip:
    # Salva o ZIP temporariamente no disco
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

    try:
        # Salva o ZIP temporariamente
        tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp_zip.write(uploaded_zip.getbuffer())
        tmp_zip.flush()
        tmp_zip.close()

        # Cria pasta temporária para extrair
        temp_dir = tempfile.mkdtemp()

        # Extrai o ZIP
        with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Agora temp_dir tem o conteúdo do ZIP, e você pode percorrer
        pastas_num = []

        for root, dirs, files in os.walk(temp_dir):
            if "pdata" in dirs:
                # pega a pasta imediatamente anterior a 'pdata'
                pasta_num = os.path.abspath(root)  # exemplo: ...\Malte_final_L\1
                pastas_num.append(pasta_num)


        espectro_tipos = {}
        erros = []

        for pasta in pastas_num:
            nome_pasta = os.path.basename(pasta)
            incluir = st.sidebar.checkbox(
                f"📁 {nome_pasta}",
                value=True,
                key=f"chk_{os.path.normpath(pasta)}"  # 🔑 chave única baseada no caminho completo
            )
            if incluir:
                espectro_tipos[pasta] = pasta

        # Botão fora do loop
        if st.sidebar.button("🚀 Carregar espectros"):
            tabs = []
            figs = []
            for pasta in espectro_tipos.values():
                # adiciona o caminho completo até 'pdata/1'
                pdata_dir = os.path.join(pasta, "pdata", "1")

                if not os.path.exists(pdata_dir):
                    erros.append(f"❌ Pasta inválida ou sem 'pdata/1': {pdata_dir}")
                    continue

                try:
                    dic, data = ng.bruker.read_pdata(pdata_dir)
                    tipo = dic.get('acqus', {}).get('PULPROG', 'Desconhecido')
                    udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
                    ndim = data.ndim

                    # --- Espectros 1D ---
                    if ndim == 1:
                        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
                        ppm = uc.ppm_scale()
                        if np.all(data == 0):
                            erros.append(f"⚠️ Espectro vazio: {pasta}")
                            continue
                        data_norm = data / np.max(np.abs(data))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ppm, y=data_norm, mode='lines',
                            name=f"{os.path.basename(pasta)} - {tipo}",
                            line=dict(width=1, color='black')
                        ))
                        fig.update_layout(
                            title=f"Espectro {tipo} - {os.path.basename(pasta)}",
                            xaxis_title="Deslocamento químico (ppm)",
                            yaxis_title="Intensidade (u.a.)",
                            xaxis=dict(autorange='reversed'),
                            template="plotly_white",
                            height=800,
                            margin=dict(l=40, r=20, t=40, b=40)
                        )
                        tabs.append(f"{os.path.basename(pasta)} ({tipo})")
                        figs.append(fig)

                    # --- Espectros 2D ---
                    elif ndim == 2:
                        uc_F1 = ng.fileiobase.uc_from_udic(udic, dim=0)
                        uc_F2 = ng.fileiobase.uc_from_udic(udic, dim=1)
                        ppm_F1 = uc_F1.ppm_scale()
                        ppm_F2 = uc_F2.ppm_scale()
                        data_log = np.log1p(np.abs(data))
                        data_norm = (data_log - np.min(data_log)) / (np.max(data_log) - np.min(data_log))
                        fig = go.Figure(data=go.Heatmap(
                            z=data_norm, x=ppm_F2, y=ppm_F1,
                            colorscale='Viridis', reversescale=True,
                            colorbar=dict(title='Intensidade (log)')
                        ))
                        fig.update_layout(
                            title=f"Espectro {tipo} - {os.path.basename(pasta)}",
                            xaxis_title="F2 (ppm)",
                            yaxis_title="F1 (ppm)",
                            xaxis=dict(autorange='reversed'),
                            yaxis=dict(autorange='reversed'),
                            template="plotly_white",
                            height=800,
                            margin=dict(l=40, r=20, t=40, b=40)
                        )
                        tabs.append(f"{os.path.basename(pasta)} ({tipo})")
                        figs.append(fig)
                    else:
                        erros.append(f"⚠️ Dimensão não suportada: {ndim}D em {pasta}")

                except Exception as e:
                    erros.append(f"Erro ao carregar {pasta}: {e}")

            # Exibe os resultados
            if figs:
                tab_objs = st.tabs(tabs)
                for tab, fig in zip(tab_objs, figs):
                    with tab:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum espectro válido encontrado.")

            if erros:
                st.error("Ocorreram alguns problemas:")
                for msg in erros:
                    st.write("•", msg)

    finally:
        # garantia de limpeza: se o temp ZIP ainda existir, remove
        try:
            if os.path.exists(tmp_zip.name):
                os.remove(tmp_zip.name)
        except Exception:
            pass
        # NOTA: temp_dir com os arquivos extraídos não é removido aqui para permitir inspeção enquanto a sessão estiver ativa.
        # Se quiser limpar ao final, descomente a linha abaixo:
        # shutil.rmtree(temp_dir, ignore_errors=True)
