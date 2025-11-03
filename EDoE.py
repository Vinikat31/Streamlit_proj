import streamlit as st
import pandas as pd
import EDoE_function as ed  # Certifique que fabi_efeito, plot_efeito, extrair_tabela_marcas, gera_design_fatorial estão aqui

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="EDoE", layout="wide")
st.title("📘 EDoE - edit design of experiment")

# =========================
# Inicializa session_state
# =========================
# O session_state armazena dados entre interações do Streamlit
if "df" not in st.session_state:
    st.session_state["df"] = None  # DataFrame principal
if "df_desing" not in st.session_state:
    st.session_state["df_desing"] = None  # Design fatorial
if "efeito" not in st.session_state:
    st.session_state["efeito"] = None  # Efeitos calculados
if "porc" not in st.session_state:
    st.session_state["porc"] = None  # Percentual dos efeitos
if "mostrar_efeito" not in st.session_state:
    st.session_state["mostrar_efeito"] = False  # Controla se a 2ª aba aparece
if "mostrar_fraci" not in st.session_state:
    st.session_state["mostrar_fraci"] = False  # Controla se a 2ª aba aparece

# =========================
# 1ª ABA — Upload do Excel
# =========================
# Expander mantém a primeira aba recolhível
with st.expander("📥 1. Selecione seu arquivo Excel (.xlsx ou .xls)", expanded=True):
    uploaded_file = st.file_uploader("", type=["xlsx", "xls"])  # Uploader de arquivo Excel

    if uploaded_file is not None:
        # Lê o Excel para DataFrame
        df = pd.read_excel(uploaded_file)

        # Extrai a tabela delimitada por '#' e '@' (função customizada)
        df = ed.extrair_tabela_marcas(df)

        # Gera o design fatorial baseado nos dados
        df_desing = ed.gera_design_fatorial(df)

        # Salva no session_state para usar em outras abas
        st.session_state["df"] = df
        st.session_state["df_desing"] = df_desing

        # Mensagem de sucesso
        st.success("✅ Arquivo carregado com sucesso!")

        # Mostra os DataFrames lado a lado

        st.markdown("### 📄 Tabela Original")
        st.table(df.fillna(""))  # Substitui NaN por vazio


        # =========================
        # Botão para calculo
        # =========================
        # Layout dos gráficos lado a lado
        col_efeito1, col_Fraci2 = st.columns(2)

        with col_efeito1:
            if st.button("Planejamento Fatorial Completo"):
                # Calcula efeitos e porcentagens usando função customizada
                efeito, porc = ed.fabi_efeito(df, df_desing)

                if efeito is not None:
                    # Armazena os resultados no session_state
                    st.session_state["efeito"] = efeito
                    st.session_state["porc"] = porc

                    # Sinaliza para mostrar a segunda aba
                    st.session_state["mostrar_efeito"] = True

                    # Feedback visual
                    st.success(f"✅ Nova aba criada com {len(efeito)} efeitos")

        with col_Fraci2:
            if st.button("Planejamento Fatorial Fracionário"):
                st.session_state["mostrar_fraci"] = True

# =========================
# 2ª ABA — Gráficos e Resultados
# =========================
# Esta seção só aparece se o botão de cálculo já foi clicado
if st.session_state["mostrar_efeito"]:
    with st.expander("2. Gráficos e Resultados do Planejamento Fatorial", expanded=True):
        # Separador visual
        st.markdown("### ⚙️ Design Fatorial Gerado")
        st.table(df_desing.fillna(""))

        # Subtítulo para os gráficos
        st.markdown("### Gráficos de Efeitos")
        # Plota os gráficos usando a função customizada
        ed.plot_efeito(st.session_state["df"], st.session_state["df_desing"])

if st.session_state["mostrar_fraci"]:
    with st.expander("2. Gráficos e Resultados do Planejamento", expanded=True):
        st.markdown("### ⚙️ Design Fatorial Gerado")


