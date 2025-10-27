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
        tmp_zip.write(uploaded_zip.getbuffer())
        tmp_zip.flush()
        tmp_zip.close()

        # Cria pasta temporária para descompactar
        temp_dir = tempfile.mkdtemp()

        # Extrai o ZIP para temp_dir
        raiz_bruker = temp_dir
        for root, dirs, files in os.walk(temp_dir):
            if "pdata" in dirs:
                raiz_bruker = os.path.dirname(root)
                break


        # Remove o arquivo zip temporário para não deixar lixo
        try:
            os.remove(tmp_zip.name)
        except Exception:
            pass

        # Lista subpastas (cada espectro é uma subpasta)
        subdirs = sorted(
            [d for d in os.listdir(raiz_bruker) if os.path.isdir(os.path.join(raiz_bruker, d))],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(ch.isdigit() for ch in x) else 0
        )

        if not subdirs:
            st.warning("Nenhuma subpasta encontrada no ZIP enviado.")
        else:
            st.sidebar.write("### Pastas encontradas:")
            espectro_tipos = {}
            erros = []

            # Cria checkboxes para cada subpasta
            for d in subdirs:
                incluir = st.sidebar.checkbox(f"📁 {d}", value=True)
                if incluir:
                    pdata_path = os.path.join(temp_dir, d, "pdata")
                    if not os.path.exists(pdata_path):
                        erros.append(f"❌ Pasta 'pdata' não encontrada em: {d}")
                        continue

                    # tenta encontrar subpastas dentro de pdata (ex: '1')
                    try:
                        subdirs_pdata = [os.path.join(pdata_path, sd) for sd in os.listdir(pdata_path)
                                         if os.path.isdir(os.path.join(pdata_path, sd))]
                    except Exception as e:
                        subdirs_pdata = []

                    if subdirs_pdata:
                        # pega a primeira subpasta dentro de pdata (normalmente '1')
                        espectro_tipos[d] = subdirs_pdata[0]
                    else:
                        erros.append(f"❌ Nenhuma subpasta dentro de 'pdata' em: {d}")

            # Botão para carregar
            if st.sidebar.button("🚀 Carregar espectros"):
                tabs = []
                figs = []

                for pasta, pdata_dir in espectro_tipos.items():
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
        # garantia de limpeza: se o temp ZIP ainda existir, remove
        try:
            if os.path.exists(tmp_zip.name):
                os.remove(tmp_zip.name)
        except Exception:
            pass
        # NOTA: temp_dir com os arquivos extraídos não é removido aqui para permitir inspeção enquanto a sessão estiver ativa.
        # Se quiser limpar ao final, descomente a linha abaixo:
        # shutil.rmtree(temp_dir, ignore_errors=True)
