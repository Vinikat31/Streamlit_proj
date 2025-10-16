import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Título do app
st.title("🎯 Visualização do Vetor de Spin em Esfera de Bloch")

# Painel lateral para controles
st.sidebar.header("Configurações do Vetor")

# Sliders para θ e φ
theta = st.sidebar.slider("Ângulo θ (graus)", 0, 180, 45, 5)
phi = st.sidebar.slider("Ângulo φ (graus)", 0, 360, 60, 5)

# Conversões
theta_r = np.radians(theta)
phi_r = np.radians(phi)

# Coordenadas do vetor
r = 0.7
z = r * np.cos(theta_r)
x = r * np.cos(phi_r) * np.sin(theta_r)
y = r * np.sin(phi_r) * np.sin(theta_r)

# Superfície esférica
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
x1 = np.outer(np.sin(u), np.sin(v))
y1 = np.outer(np.sin(u), np.cos(v))
z1 = np.outer(np.cos(u), np.ones_like(v))

# Criação do gráfico
fig = go.Figure()

# Superfície esférica (transparente)
fig.add_trace(go.Surface(
    x=x1, y=y1, z=z1,
    colorscale='Greys',
    opacity=0.05,
    showscale=False
))

# Eixos
fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1],
                           mode='lines', line=dict(color='black', width=4)))
fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0],
                           mode='lines', line=dict(color='black', width=4)))
fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0],
                           mode='lines', line=dict(color='black', width=4)))

# Vetor principal
fig.add_trace(go.Scatter3d(
    x=[0, x], y=[0, y], z=[0, z],
    mode='lines',
    line=dict(color='red', width=6),
    marker=dict(size=5, color='red')
))

# Vetor como cone (seta)
fig.add_trace(go.Cone(
    x=[x], y=[y], z=[z],
    u=[x], v=[y], w=[z],
    sizemode="scaled",
    sizeref=0.2,
    colorscale="Reds",
    showscale=False
))

# Layout do gráfico
fig.update_layout(
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
        annotations=[
            dict(showarrow=False, x=0, y=0, z=1, text="|α⟩", font=dict(size=12, color="black")),
            dict(showarrow=False, x=0, y=0, z=-1, text="|β⟩", font=dict(size=12, color="black")),
            dict(showarrow=False, x=1, y=0, z=0, text="(|α⟩+|β⟩)/√2", font=dict(size=12, color="black")),
            dict(showarrow=False, x=0, y=1, z=0, text="(|α⟩+i|β⟩)/√2", font=dict(size=12, color="black"))
        ]
    )
)

# Exibir gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)
