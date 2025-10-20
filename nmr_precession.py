import streamlit as st
import spin_graph as spg
import pulse_sequence as ps



# --- Inputs do usuário ---
st.sidebar.header("NMR Spin Simulator")
theta = st.sidebar.number_input("θ (°)", min_value=0, max_value=180, value=0, step=10)
phi = st.sidebar.number_input("φ (°)", min_value=0, max_value=360, value=0, step=10)
wnut = st.sidebar.number_input("Pulse amplitude (ω₁) [Hz]", min_value=0.0, value=500.0, step=10.0)
phi_p = st.sidebar.number_input("Phase of pulse (Φ₁) [°]", min_value=0.0, max_value=360.0, value=0.0, step=5.0)
omega_ref = st.sidebar.number_input("Rotating frame ω₀ [MHz]", min_value=2.0, value=2.0, step=2.0)
tau = st.sidebar.number_input("Duration of pulse [ms]", min_value=0.0, value=0.50, step=0.10)
time_relax = st.sidebar.number_input("Relaxation time [ms]", min_value=0.0, value=1.0, step=1.0)





# --- Gráfico principal ---
fig_main = spg.update_graph(theta, phi, omega_ref, wnut, phi_p, tau, time_relax)

# --- Gráfico secundário (inset) ---
# Crie uma nova função no seu módulo `spin_graph.py` que retorna o segundo gráfico
fig_inset = ps.inset_graph(theta, phi, omega_ref, wnut, phi_p, tau, time_relax)

# --- Inserir ambos no Streamlit ---
config = {
    "displayModeBar": False,  # Oculta a barra de ferramentas
    "responsive": True         # Faz o gráfico se ajustar à largura do container
}

st.plotly_chart(fig_inset, use_container_width=False, config=config)

st.plotly_chart(fig_main, use_container_width=False, config=config)


