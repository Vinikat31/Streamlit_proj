import streamlit as st
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import os
import tempfile
import zipfile
import shutil

st.set_page_config(page_title="NMR proc", layout="wide")
st.title("🧪 NMR proc")

# --- Upload do ZIP da pasta raiz ---
uploaded_zip = st.sidebar.file_uploader(
    "📤 Envie a pasta raiz dos espectros (compactada em ZIP)",
    type="zip"
)

def extract_number(s):
    """Extrai números do nome da pasta para ordenar corretamente."""
    nums = ''.join(filter(str.isdigit, s))
    return int(nums) if nums else float('inf')

def encontrar_pdata(root_dir):
    """
    Procura recursivamente todas as pastas 'pdata' dentro de root_dir.
    Retorna um dicionário {nome_da_subpasta: caminho_para_pdata}.
    """
    pdata_dict = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "pdata" in dirnames:
            pasta_nome = os.path.basename(dirpath)
            pdata_dict[pasta_nome] = os.path.join(dirpath, "pdata")
    return pdata_dict

if uploaded_zip:
    # Salva o ZIP temporariamente no disco
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_zip.write(uploaded_zip.getbuffer())
    tmp_zip.flush()
    tmp_zip.close()

    # Cria pasta temporária para descompactar
    temp_dir = tempfile.mkdtemp()

    try:
        # Extrai o ZIP para temp_dir
        with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # --- Remove o ZIP agora ---
        try:
            os.remove(tmp_zip.name)
        except Exception:
            pass

        # Procura todas as pastas 'pdata'
        espectro_tipos = encontrar_pdata(temp_dir)
        espectro_tipos = dict(sorted(espectro_tipos.items(), key=lambda x: extract_number(x[0])))

        if not espectro_tipos:
            st.warning("Nenhuma pasta 'pdata' encontrada no ZIP enviado.")
        else:
            st.sidebar.write("### Pastas 'pdata' encontradas:")
            espectro_tipos = {}
            checkboxes = {}
            erros = []

            # Cria checkboxes para cada pasta encontrada
            for pasta, pdata_dir in espectro_tipos.items():
                checkboxes[pasta] = st.sidebar.checkbox(f"📁 {pasta}", value=True)
                if checkboxes:
                    pdata_path = os.path.join(temp_dir, pasta, "pdata")
                    if not os.path.exists(pdata_path):
                        erros.append(f"❌ Pasta 'pdata' não encontrada em: {pasta}")
                        continue

                    # tenta encontrar subpastas dentro de pdata (ex: '1')
                    try:
                        subdirs_pdata = [os.path.join(pdata_path, sd) for sd in os.listdir(pdata_path)
                                         if os.path.isdir(os.path.join(pdata_path, sd))]
                    except Exception as e:
                        subdirs_pdata = []

                    if subdirs_pdata:
                        # pega a primeira subpasta dentro de pdata (normalmente '1')
                        espectro_tipos[pasta] = subdirs_pdata[0]
                    else:
                        erros.append(f"❌ Nenhuma subpasta dentro de 'pdata' em: {pasta}")

            # Botão para carregar
            if st.sidebar.button("🚀 Carregar espectros"):
                tabs = []
                figs = []

                for pasta, pdata_dir in espectro_tipos.items():
                    if not checkboxes[pasta]:
                        continue
                    if not os.path.exists(pdata_dir):
                        erros.append(f"❌ Pasta inválida: {pasta}")
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
                                name=f"{pasta} - {tipo}", line=dict(width=1, color='black')
                            ))
                            fig.update_layout(
                                title=f"Espectro {tipo} - {pasta}",
                                xaxis_title="Deslocamento químico (ppm)",
                                yaxis_title="Intensidade (u.a.)",
                                xaxis=dict(autorange='reversed'),
                                template="plotly_white",
                                height=800,
                                margin=dict(l=40, r=20, t=40, b=40)
                            )
                            tabs.append(f"{pasta} ({tipo})")
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
                                title=f"Espectro {tipo} - {pasta}",
                                xaxis_title="F2 (ppm)",
                                yaxis_title="F1 (ppm)",
                                xaxis=dict(autorange='reversed'),
                                yaxis=dict(autorange='reversed'),
                                template="plotly_white",
                                height=800,
                                margin=dict(l=40, r=20, t=40, b=40)
                            )
                            tabs.append(f"{pasta} ({tipo})")
                            figs.append(fig)
                        else:
                            erros.append(f"⚠️ Dimensão não suportada: {ndim}D em {pasta}")

                    except Exception as e:
                        erros.append(f"Erro ao carregar {pasta}: {e}")

                # Exibe resultados
                if figs:
                    tab_objs = st.tabs(tabs)
                    for tab, fig in zip(tab_objs, figs):
                        with tab:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Nenhum espectro válido encontrado.")

                # Mostra erros
                if erros:
                    st.error("Ocorreram alguns problemas:")
                    for msg in erros:
                        st.write("•", msg)

    finally:
        # NOTA: mantém temp_dir durante a sessão para inspeção, se quiser limpar:
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass
