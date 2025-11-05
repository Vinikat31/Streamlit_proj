# ==========================================
# 📘 app.py — Painel principal do EDoE
# ==========================================
import streamlit as st
import sys
import os

# Configuração inicial da página
st.set_page_config(page_title="Painel Experimental EDoE", layout="wide")

# Caminho para o módulo de funções e interfaces
sys.path.append(os.path.join(os.path.dirname(__file__), "functions"))
sys.path.append(os.path.join(os.path.dirname(__file__), "interfaces"))

# Importa a interface do planejamento fatorial
import fatorial_completo as fatorial

# ==========================================
# 🧭 Barra lateral
# ==========================================
st.sidebar.title("🧭 Menu de Navegação")
pagina = st.sidebar.radio(
    "Selecione o tipo de planejamento:",
    [
        "🏠 Página Inicial",
        "📊 Planejamento Fatorial Completo",
        "🧮 Outros Planejamentos"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Desenvolvido por:  "
    "**Vinicius S. Ferreira**  "
    "**Dr. Dennis Ferreira**  "
    "**Prof. Dr. Edenir R. P. Filho**"
)


# ==========================================
# 📄 Conteúdo das páginas
# ==========================================
if pagina == "🏠 Página Inicial":
    st.title("🧪 Painel Experimental — EDoE")
    st.markdown("""
    ### Bem-vindo ao Painel de Planejamento Experimental

    Este aplicativo foi desenvolvido para facilitar a **análise de planejamentos fatoriais**
    e outros métodos de **Design of Experiments (DoE)**, para as aulas de Quimiometria do Prof. Dr. Edenir R. P. Filho.

    ---
    **⬅ Selecione o tipo de planejamento na barra lateral**
    """)

elif pagina == "📊 Planejamento Fatorial Completo":
    fatorial.interface_fatorial()

elif pagina == "🧮 Outros Planejamentos":
    st.title("🧮 Outros Planejamentos")
    st.markdown("""
    🔧 Esta seção será utilizada futuramente para incluir novos planejamentos,
    como:
    - Plackett-Burman
    - Central Composto Rotacional (CCR)
    - Box-Behnken  

    *Em breve...*
    """)

