import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# --- Constantes ---
MHZ_TO_HZ = 1e6
RAD_TO_DEG = np.pi / 180

# --- Superfície esférica ---
u = np.linspace(0, np.pi, 10)
v = np.linspace(0, 2 * np.pi, 15)
x1 = np.outer(np.sin(u), np.sin(v))
y1 = np.outer(np.sin(u), np.cos(v))
z1 = np.outer(np.cos(u), np.ones_like(v))

st.title("Simulação NMR - Streamlit")

# --- Inputs do usuário ---
st.sidebar.header("Parâmetros da simulação")
theta = st.sidebar.number_input("θ (°)", min_value=0, max_value=180, value=0, step=10)
phi = st.sidebar.number_input("φ (°)", min_value=0, max_value=360, value=0, step=10)
wnut = st.sidebar.number_input("Pulse amplitude (ω₁) [Hz]", min_value=0.0, value=500.0, step=10.0)
phi_p = st.sidebar.number_input("Phase of pulse (Φ₁) [°]", min_value=0.0, max_value=360.0, value=0.0, step=5.0)
omega_ref = st.sidebar.number_input("Rotating frame ω₀ [MHz]", min_value=0.0, value=720.0, step=2.0)
tau = st.sidebar.number_input("Duration of pulse [s]", min_value=0.0, value=0.0005, step=0.00001)
time_relax = st.sidebar.number_input("Relaxation time [s]", min_value=0.0, value=50.0, step=50.0)

# --- Conversão de unidades ---
theta_r = theta * RAD_TO_DEG
phi_r = phi * RAD_TO_DEG
omega_ref = omega_ref * 2 * np.pi * MHZ_TO_HZ
theta_pulse = wnut * (90/500) * RAD_TO_DEG
wnut = wnut * 2 * np.pi
phi_pulse = phi_p * RAD_TO_DEG + (90 * RAD_TO_DEG)

# --- Inicialização do estado ---
r = 0.7
x, y, z = r * np.sin(theta_r) * np.cos(phi_r), r * np.sin(theta_r) * np.sin(phi_r), r * np.cos(theta_r)
xp, yp, zp = r * np.sin(theta_pulse) * np.cos(phi_pulse), r * np.sin(theta_pulse) * np.sin(phi_pulse), r * np.cos(theta_pulse)
psi_0 = np.array([[np.cos(theta / 2)], [np.sin(theta / 2) * np.exp(1j * phi)]])
zlist, xlist, ylist = [np.cos(theta)], [np.cos(phi) * np.sin(theta)], [np.sin(phi) * np.sin(theta)]
tlist = np.linspace(0, tau, 200)
psi_tlist = [psi_0]
delta_t = tlist[1] - tlist[0]

def calculate_angles(psi_t):
    A, B = psi_t[0], psi_t[1]
    c1, c2 = np.real(A[0]), np.imag(A[0])
    c3, c4 = np.real(B[0]), np.imag(B[0])
    def func(x):
        return [
            np.cos(x[0]) * np.cos(x[1]) - c1,
            np.cos(x[0]) * np.sin(x[1]) - c2,
            np.sin(x[0]) * np.cos(x[2]) - c3,
            np.sin(x[0]) * np.sin(x[2]) - c4,
        ]
    return fsolve(func, [1, 1, 1, 1], xtol=1e-8, maxfev=1000000, factor=0.5)

# --- Evolução temporal ---
for i in range(len(tlist)-1):
    beta = wnut * delta_t
    Ru_beta = np.array([
        [np.cos(beta/2), -1j*np.sin(beta/2)*np.exp(-1j*phi_p)],
        [-1j*np.sin(beta/2)*np.exp(1j*phi_p), np.cos(beta/2)]
    ])
    alpha = omega_ref * tlist[i+1]
    Rz = np.array([[np.exp(-1j*alpha/2),0],[0,np.exp(1j*alpha/2)]])
    alpha_0 = -omega_ref * tlist[i]
    Rz_0 = np.array([[np.exp(-1j*alpha_0/2),0],[0,np.exp(1j*alpha_0/2)]])
    P1 = Rz_0.dot(psi_tlist[i])
    R2 = Ru_beta.dot(P1)
    psi_t = Rz.dot(R2)
    psi_tlist.append(psi_t)

    root = calculate_angles(psi_t)
    theta_pp = root[0]*2
    phi_pp = -(root[1]-root[2])
    znew = np.cos(theta_pp)
    xnew = np.cos(phi_pp)*np.sin(theta_pp)
    ynew = np.sin(phi_pp)*np.sin(theta_pp)
    zlist.append(znew)
    xlist.append(xnew)
    ylist.append(ynew)

# --- Gráfico Plotly ---
fig = go.Figure()

# Superfície esférica
fig.add_trace(go.Surface(x=x1, y=y1, z=z1, colorscale='Greys', opacity=0.05, showscale=False))
fig.add_trace(go.Surface(x=x1*0.2, y=y1*0.2, z=z1*0.2, colorscale='Reds', showscale=False))

# Trajetória do spin
fig.add_trace(go.Scatter3d(x=xlist, y=ylist, z=zlist, mode='lines', line=dict(color='red', width=4)))

# Estado inicial/final
fig.add_trace(go.Scatter3d(x=[xlist[0]], y=[ylist[0]], z=[zlist[0]], mode='markers', marker=dict(color='red', size=5), name='Initial State'))
fig.add_trace(go.Scatter3d(x=[xlist[-1]], y=[ylist[-1]], z=[zlist[-1]], mode='markers', marker=dict(color='blue', size=5), name='Final State'))

fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
        camera=dict(eye=dict(x=1, y=1, z=1))
    ),
    title="Spin Animation",
    height=600,
    margin=dict(l=0,r=0,t=50,b=0)
)

st.plotly_chart(fig, use_container_width=True)
