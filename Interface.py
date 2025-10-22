import streamlit as st
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="Visualizador de RMN",
    layout="wide"
)
st.title("🧪 Visualizador de Espectros de RMN")

# --- Selecionar pasta principal ---
root_dir = st.sidebar.text_input("📂 Informe o caminho da pasta principal:")

# --- Verifica se o diretório é válido ---
if root_dir and os.path.isdir(root_dir):
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if not subdirs:
        st.sidebar.warning("Nenhuma subpasta encontrada nesse diretório.")
    else:
        st.sidebar.write("### Pastas encontradas:")
        espectro_tipos = {}
        for d in subdirs:
            incluir = st.sidebar.checkbox(f"📁 {d}", value=True)
            espectro_tipos[d] = tipo = 1

        # --- Botão para carregar ---
        if st.sidebar.button("🚀 Carregar espectros"):
            tabs = []
            figs = []
            erros = []

            for pasta, tipo in espectro_tipos.items():

                pdata_dir = os.path.join(root_dir, pasta, "pdata", "1")
                if not os.path.exists(pdata_dir):
                    erros.append(f"❌ Pasta inválida: {pasta}")
                    continue

                try:
                    dic, data = ng.bruker.read_pdata(pdata_dir)
                    tipo = dic.get('acqus', {}).get('PULPROG', 'Desconhecido')
                    udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
                    ndim = data.ndim

                    # --- Espectros 1D (¹H, ¹³C) ---
                    if ndim == 1:
                        uc = ng.fileiobase.uc_from_udic(udic, dim=0)
                        ppm = uc.ppm_scale()

                        # Evita espectros vazios
                        if np.all(data == 0):
                            erros.append(f"⚠️ Espectro vazio em {pasta}")
                            continue

                        # Normalização
                        data_norm = data / np.max(np.abs(data))

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ppm,
                            y=data_norm,
                            mode='lines',
                            name=f"{pasta} - {tipo}",
                            line=dict(width=1, color='black')
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

                    # --- Espectros 2D (COSY, HSQC, HMBC) ---
                    elif ndim == 2:
                        uc_F1 = ng.fileiobase.uc_from_udic(udic, dim=0)
                        uc_F2 = ng.fileiobase.uc_from_udic(udic, dim=1)
                        ppm_F1 = uc_F1.ppm_scale()
                        ppm_F2 = uc_F2.ppm_scale()

                        # Normalização logarítmica
                        data_log = np.log1p(np.abs(data))
                        data_norm = (data_log - np.min(data_log)) / (np.max(data_log) - np.min(data_log))

                        fig = go.Figure(data=go.Heatmap(
                            z=data_norm,
                            x=ppm_F2,
                            y=ppm_F1,
                            colorscale='Viridis',
                            reversescale=True,
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

            # --- Exibe resultados ---
            if figs:
                tab_objs = st.tabs(tabs)
                for tab, fig in zip(tab_objs, figs):
                    with tab:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Nenhum espectro válido encontrado.")

            # --- Mostra mensagens de erro ---
            if erros:
                st.error("Ocorreram alguns problemas:")
                for msg in erros:
                    st.write("•", msg)

else:
    st.sidebar.info("Digite o caminho de uma pasta válida para começar.")
