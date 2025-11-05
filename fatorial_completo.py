# ==========================================
# 📊 interfaces/fatorial_completo.py
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Caminho para funções
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "functions"))
import EDoE_function as ed


def interface_fatorial():
    st.title("📊 Planejamento Fatorial Completo")

    # --- Estado inicial ---
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "efeito" not in st.session_state:
        st.session_state["efeito"] = None
    if "porc" not in st.session_state:
        st.session_state["porc"] = None

    # --- Upload de arquivo ---
    with st.expander("📥 1. Selecione seu arquivo Excel (.xlsx ou .xls)", expanded=True):
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"])

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            df = ed.extrair_tabela_marcas(df)
            df_desing = ed.gera_design_fatorial(df)
            df_display = df.fillna("")

            st.session_state["df"] = df_display
            st.session_state["df_desing"] = df_desing

            st.success("✅ Arquivo carregado com sucesso!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📄 Tabela Original")
                st.dataframe(df_display, use_container_width=True)
            with col2:
                st.markdown("### ⚙️ Design Fatorial Gerado")
                st.dataframe(df_desing, use_container_width=True)

                if st.button("📊 Calcular Efeito Fabi"):
                    efeito, porc, erro_efeito, t_val = ed.fabi_efeito(df, df_desing)

                    if efeito is not None:
                        st.session_state["efeito"] = efeito
                        st.session_state["porc"] = porc
                        st.session_state["erro_efeito"] = erro_efeito
                        st.session_state["t_val"] = t_val

                        msg = f"✅ Nova tabela criada ({len(efeito)} efeitos)"
                        if erro_efeito is not None and t_val is not None:
                            msg += f" | Erro = {erro_efeito:.4f}, t = {t_val:.4f}"

                        st.success(msg)

    # --- Resultados e gráficos ---
    if st.session_state.get("efeito") is not None:
        with st.expander("📈 2. Resultados e Gráficos", expanded=True):
            st.write("### 🔍 Análise dos Efeitos")
            erro_efeito_val = st.session_state.get("erro_efeito", 1.0)
            t_val = st.session_state.get("t_val", 1.96)

            ed.plot_efeito(
                df=st.session_state["df"],
                df_desing=st.session_state["df_desing"],
                erro_efeito_val=erro_efeito_val,
                t_val=t_val
            )


