# ==========================================
# 🧠 Importação de bibliotecas
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import EDoE_function as ed  # Arquivo com as funções auxiliares

# ==========================================
# ⚙️ Configuração inicial da página
# ==========================================
st.set_page_config(page_title="Visualizador Excel Interativo", layout="wide")
st.title("📘 Visualizador Interativo de Arquivo Excel")

# ==========================================
# 🔁 Inicialização do estado da sessão
# ==========================================
# Isso garante que os dados não se percam quando o Streamlit recarregar a interface
if "df" not in st.session_state:
    st.session_state["df"] = None
if "efeito" not in st.session_state:
    st.session_state["efeito"] = None
if "porc" not in st.session_state:
    st.session_state["porc"] = None

# ==========================================
# 📥 1ª SEÇÃO — Upload do arquivo Excel
# ==========================================
with st.expander("📥 1. Selecione seu arquivo Excel (.xlsx ou .xls)", expanded=True):

    # Upload do arquivo pelo usuário
    uploaded_file = st.file_uploader("", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Lê o Excel enviado
        df = pd.read_excel(uploaded_file)

        # Extrai a tabela delimitada por '#' e '@' usando função personalizada
        df = ed.extrair_tabela_marcas(df)

        # Gera automaticamente o design fatorial correspondente
        df_desing = ed.gera_design_fatorial(df)

        # Substitui valores NaN por string vazia (para melhor exibição)
        df_display = df.fillna("")

        # Armazena as tabelas no estado da sessão
        st.session_state["df"] = df_display
        st.session_state["df_desing"] = df_desing

        # Mensagem de sucesso
        st.success("✅ Arquivo carregado com sucesso!")

        # Mostra as tabelas lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📄 Tabela Original")
            st.dataframe(df_display, use_container_width=True)
        with col2:
            st.markdown("### ⚙️ Design Fatorial Gerado")
            st.dataframe(df_desing, use_container_width=True)

        # Botão para calcular os efeitos (Efeito Fabi)
        if st.button("📊 Calcular Efeito Fabi"):
            efeito, porc = ed.fabi_efeito(df, df_desing)

            if efeito is not None:
                # Salva os resultados no session_state
                st.session_state["efeito"] = efeito
                st.session_state["porc"] = porc

                # Confirmação visual
                st.success(f"✅ Nova tabela criada ({len(efeito)} efeitos)")


# ==========================================
# 📊 2ª SEÇÃO — Exibir resultados e gráficos
# ==========================================
# Essa seção é carregada apenas se os efeitos já foram calculados
if st.session_state.get("efeito") is not None:
    with st.expander("📈 2. Resultados e Gráficos", expanded=True):

        # Entradas para o erro e o valor t
        col1, col2 = st.columns(2)
        with col1:
            erro_efeito_val = st.number_input(
                "⚠️ Valor de erro do efeito",
                min_value=0.0,
                value=1.0,
                step=0.5
            )
        with col2:
            t_val = st.number_input(
                "🧮 Valor de t",
                min_value=0.0,
                value=0.95,
                step=0.05
            )

        # Exibe os gráficos de efeitos com base nas funções definidas em EDoE_function
        st.write("### 🔍 Análise dos Efeitos")
        ed.plot_efeito(
            st.session_state["df"],
            st.session_state["df_desing"],
            erro_efeito_val=erro_efeito_val,
            t_val=t_val
        )