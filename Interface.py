import streamlit as st
import os
import interface_f

st.set_page_config(page_title="Visualizador de RMN", layout="wide")
st.title("🧪 Visualizador de Espectros de RMN")

# --- Upload ZIP ou arquivo individual ---
uploaded_zip = st.sidebar.file_uploader(
    "📤 Envie a pasta raiz dos espectros (ZIP)",
    type="zip"
)

uploaded_file = st.sidebar.file_uploader(
    "📤 Ou envie um arquivo individual (.fid ou pasta compactada de um espectro)",
    type=["fid"]
)

folder_path = None

if uploaded_zip:
    zip_path = interface_f.salvar_zip_temporario(uploaded_zip)
    folder_path = interface_f.extrair_zip(zip_path)

elif uploaded_file:
    folder_path = interface_f.salvar_arquivo_temporario(uploaded_file)

if folder_path:
    pastas_num = interface_f.encontrar_pastas_com_pdata(folder_path)

    espectro_tipos = {}
    for pasta in pastas_num:
        nome_pasta = os.path.basename(pasta)
        incluir = st.sidebar.checkbox(
            f"📁 {nome_pasta}", value=True,
            key=f"chk_{os.path.normpath(pasta)}"
        )
        if incluir:
            espectro_tipos[pasta] = pasta

    if st.sidebar.button("🚀 Carregar espectros"):
        tabs, figs, erros = interface_f.processar_pastas(espectro_tipos.values())

        if uploaded_zip:
            try:
                os.remove(zip_path)
                st.sidebar.success("🗑️ Arquivo ZIP temporário removido.")
            except:
                pass

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
