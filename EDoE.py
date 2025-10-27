import streamlit as st
import pandas as pd
import numpy as np
import EDoE_function as ed  # Certifique-se que fabi_efeito e plot_efeito estão aqui

st.set_page_config(page_title="Visualizador Excel Interativo", layout="wide")
st.title("📘 Visualizador Interativo de Arquivo Excel")

# Inicializa session_state
if "df" not in st.session_state:
    st.session_state["df"] = None
if "efeito" not in st.session_state:
    st.session_state["efeito"] = None
if "porc" not in st.session_state:
    st.session_state["porc"] = None

# ==========================================
# 1ª ABA — Upload do Excel
# ==========================================
with st.expander("📥 1. Selecione seu arquivo Excel (.xlsx ou .xls)", expanded=True):
    uploaded_file = st.file_uploader("", type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Extrai a tabela delimitada por '#' e '@'
        df = ed.extrair_tabela_marcas(df)

        # Gera o design fatorial
        df_desing = ed.gera_design_fatorial(df)

        # Substitui NaN para exibição
        df_display = df.fillna("")

        # Armazena no estado
        st.session_state["df"] = df_display
        st.session_state["df_desing"] = df_desing

        st.success("✅ Arquivo carregado com sucesso!")

        # Mostrar DataFrames lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📄 Tabela Original")
            st.dataframe(df_display, use_container_width=True)
        with col2:
            st.markdown("### ⚙️ Design Fatorial Gerado")
            st.dataframe(df_desing, use_container_width=True)

        # Botão para calcular efeito
        if st.button("Efeito Fabi"):
            efeito, porc = ed.fabi_efeito(df, df_desing)
            if efeito is not None:
                st.session_state["efeito"] = efeito
                st.session_state["porc"] = porc
                st.success(f"✅ Nova tabela criada ({len(efeito)} efeitos)")




# ==========================================
# 2ª ABA — Exibir nova tabela e gráficos
# ==========================================
if st.session_state.get("efeito") is not None:
    with st.expander("📋 3. Resultados e Gráficos", expanded=True):
        st.write("### Efeito")
        ed.plot_efeito(st.session_state["efeito"], st.session_state["porc"], erro_efeito=2, t=0.05)

