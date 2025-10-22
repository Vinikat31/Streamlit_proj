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

def update_graph(theta, phi, omega_ref, wnut, phi_p, tau, time):
    """Update the 3D graph based on user inputs."""

    # Input validation
    if not (0 <= theta <= 180):
        raise ValueError("Theta must be between 0 and 180 degrees.")
    if not (0 <= phi < 360):
        raise ValueError("Phi must be between 0 and 360 degrees.")
    if omega_ref <= 0:
        raise ValueError("Rotating frame frequency must be positive.")
    if wnut <= 0:
        raise ValueError("Pulse amplitude must be positive.")
    if tau <= 0:
        raise ValueError("Duration of pulse must be positive.")

    # Convert inputs
    tau = float(tau)/1000
    theta_r= theta = np.radians(theta)
    phi_r = phi = np.radians(phi)
    omega_ref = float(omega_ref) * 2 * np.pi * MHZ_TO_HZ
    theta_pulse = float(wnut*(90/500)) * RAD_TO_DEG
    wnut = float(wnut) * 2 * np.pi
    phi_p = float(phi_p) * RAD_TO_DEG
    phi_pulse = phi_p + (90) * RAD_TO_DEG


    r = 0.7
    x, y, z = r * np.sin(theta_r) * np.cos(phi_r), r * np.sin(theta_r) * np.sin(phi_r), r * np.cos(theta_r)
    xp, yp, zp = r * np.sin(theta_pulse) * np.cos(phi_pulse), r * np.sin(theta_pulse) * np.sin(phi_pulse), r * np.cos(
        theta_pulse)
    # Initial state
    psi_0 = np.array([[np.cos(theta / 2)], [np.sin(theta / 2) * np.exp(1j * phi)]])
    zlist, xlist, ylist = [np.cos(theta)], [np.cos(phi) * np.sin(theta)], [np.sin(phi) * np.sin(theta)]
    tlist = np.linspace(0, tau, 200)
    psi_tlist = [psi_0]
    delta_t = tlist[1] - tlist[0]

    # Function to calculate angles
    def calculate_angles(psi_t):
        A, B = psi_t[0], psi_t [1]
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

    for i in range(len(tlist) - 1):
        beta = wnut * delta_t
        Ru_beta = np.array([
            [np.cos(beta / 2), -1j * np.sin(beta / 2) * np.exp(-1j * phi_p)],
            [-1j * np.sin(beta / 2) * np.exp(1j * phi_p), np.cos(beta / 2)],
        ])

        alpha = omega_ref * tlist[i + 1]
        Rz = np.array([[np.exp(-1j * alpha / 2), 0], [0, np.exp(1j * alpha / 2)]])
        alpha_0 = -omega_ref * tlist[i]
        Rz_0 = np.array([[np.exp(-1j * alpha_0 / 2), 0], [0, np.exp(1j * alpha_0 / 2)]])
        P1 = Rz_0.dot(psi_tlist[i])
        R2 = Ru_beta.dot(P1)
        psi_t = Rz.dot(R2)
        psi_tlist.append(psi_t)

        root = calculate_angles(psi_t)
        theta_pp = root[0] * 2
        phi_pp = -(root[1] - root[2])

        znew = np.cos(theta_pp)
        xnew = np.cos(phi_pp) * np.sin(theta_pp)
        ynew = np.sin(phi_pp) * np.sin(theta_pp)

        zlist.append(znew)
        xlist.append(xnew)
        ylist.append(ynew)



    # Create the visualization with Plotly
    fig = go.Figure()
    # Superfície esférica
    fig.add_trace(go.Surface(
        x=x1*0.2, y=y1*0.2, z=z1*0.2,
        colorscale='Reds',
        showscale=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0.85, 0],
        mode='lines',
        line=dict(color='green', width=10),
        marker=dict(size=5, color='red'),
        name="B0",
        opacity=0.5
    ))
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0.85],
        u=[0], v=[0], w=[1],
        sizemode="scaled",
        sizeref=0.2,
        colorscale="greens",
        showscale=False,
        opacity=0.5
    ))
    fig.add_trace(go.Scatter3d(
        x=[-x, x], y=[-y, y], z=[-z, z],
        mode='lines',
        line=dict(color='red', width=10),
        marker=dict(size=5, color='red'),
        name="Spin"
    ))
    fig.add_trace(go.Cone(
        x=[x], y=[y], z=[z],
        u=[x], v=[y], w=[z],
        sizemode="scaled",
        sizeref=0.2,
        colorscale="Reds",
        showscale=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[xp, 0], y=[yp, 0], z=[-zp, 0],
        mode='lines',
        line=dict(color='blue', width=10),
        marker=dict(size=5, color='red'),
        name="Pulse"
    ))
    fig.add_trace(go.Cone(
        x=[xp * 0.4], y=[yp * 0.4], z=[-zp * 0.4],
        u=[-xp], v=[-yp], w=[zp],
        sizemode="scaled",
        sizeref=0.2,
        colorscale="blues",
        showscale=False
    ))

    # Coordinate axes
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines',name='Vector',legendgroup='grupo_vector',showlegend=True, line=dict(color='black', width=4)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', legendgroup='grupo_vector', showlegend=False, line=dict(color='black', width=4)))
    fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', legendgroup='grupo_vector', showlegend=False, line=dict(color='black', width=4)))

    # Trajectory
    fig.add_trace(go.Scatter3d(x=xlist, y=ylist, z=zlist, mode='lines',name='precession', line=dict(color='gray', width=2)))

    # Initial and final states
    fig.add_trace(go.Scatter3d(x=[xlist[0]], y=[ylist[0]], z=[zlist[0]], mode='markers', marker=dict(color='red', size=5), name='Initial State'))
    fig.add_trace(go.Scatter3d(x=[xlist[-1]], y=[ylist[-1]], z=[zlist[-1]], mode='markers', marker=dict(color='blue', size=5), name='Final State'))

    # Trajectory and animation
    frames = []

    # Invertendo as listas
    xlist = xlist[::-1]  # Inverte xlist
    ylist = ylist[::-1]  # Inverte ylist
    zlist = zlist[::-1]  # Inverte zlist
    for i in range(len(tlist)):
        frames.append(go.Frame(
            data=[
                # Atualizar o vetor spin
                go.Cone(
                    x=[xlist[i]*0.7], y=[ylist[i]*0.7], z=[zlist[i]*0.7],
                    u=[xlist[i]*0.7], v=[ylist[i]*0.7], w=[zlist[i]*0.7],
                    sizemode="scaled",
                    sizeref=0.2,
                    colorscale="Reds",
                    showscale=False,
                    opacity = 1.0
                ),
                go.Scatter3d(
                    x=[-xlist[i]*0.7, xlist[i]*0.7], y=[-ylist[i]*0.7, ylist[i]*0.7], z=[-zlist[i]*0.7, zlist[i]*0.7],
                    mode='lines',
                    line=dict(color='red', width=10),
                    marker=dict(size=5, color='red'),
                    name="Spin"
                ),
                # Linha da trajetória até o ponto atual
                go.Scatter3d(
                    x=xlist[:i + 1], y=ylist[:i + 1], z=zlist[:i + 1],
                    mode='lines',
                    line=dict(color='red', width=10)
                ),
                # Superfície esférica
                go.Surface(
                    x=x1 * 0.2, y=y1 * 0.2, z=z1 * 0.2,
                    colorscale='Reds',
                    showscale=False
                )

            ],
            name=f"frame_{i}"
        ))

    # Adicionar os frames à figura
    fig.frames = frames

    # Configuração de botões de reprodução
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, dict(frame=dict(duration=time/10, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")])
                ]
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=1, y=1, z=1)),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        title="Spin Animation",
        margin=dict(l=0, r=0, t=0, b=0)
    )


    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title=""),
            annotations=[
                dict(
                    showarrow=False,
                    x=0,  # Posição no eixo X
                    y=0,  # Posição no eixo Y
                    z=1,  # Posição no eixo Z
                    text='||\u03B1>',  # Texto a ser exibido
                    font=dict(size=12, color="grey"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="bottom"  # Alinha o texto na parte inferior no eixo Y
                ),
                dict(
                    showarrow=False,
                    x=0,  # Posição no eixo X
                    y=0,  # Posição no eixo Y
                    z=-1,  # Posição no eixo Z
                    text='||\u03B2>',  # Texto a ser exibido
                    font=dict(size=12, color="grey"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="top"  # Alinha o texto na parte inferior no eixo Y
                ),
                dict(
                    showarrow=False,
                    x=0,  # Posição no eixo X
                    y=1,  # Posição no eixo Y
                    z=0,  # Posição no eixo Z
                    text="|\u03B1>+i|\u03B2>\\√2",  # Texto a ser exibido
                    font=dict(size=12, color="grey"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="top"  # Alinha o texto na parte inferior no eixo Y
                ),
                dict(
                    showarrow=False,
                    x=1,  # Posição no eixo X
                    y=0,  # Posição no eixo Y
                    z=0,  # Posição no eixo Z
                    text="|\u03B1>+|\u03B2>\\√2",  # Texto a ser exibido
                    font=dict(size=12, color="grey"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="top"  # Alinha o texto na parte inferior no eixo Y
                ),
                dict(
                    showarrow=False,
                    x=xp * 0.6,  # Posição no eixo X
                    y=yp * 0.6,  # Posição no eixo Y
                    z=zp * 0.6,  # Posição no eixo Z
                    text="Pulse",  # Texto a ser exibido
                    font=dict(size=15, color="blue"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="top",  # Alinha o texto na parte inferior no eixo Y

                ),
                dict(
                    showarrow=False,
                    x=0.1,  # Posição no eixo X
                    y=0,  # Posição no eixo Y
                    z=0.9,  # Posição no eixo Z
                    text="B0",  # Texto a ser exibido
                    font=dict(size=15, color="green"),
                    xanchor="center",  # Alinha o texto no centro no eixo X
                    yanchor="top"  # Alinha o texto na parte inferior no eixo Y
                )
            ]
        ),

        width=1000,  # largura do gráfico
        height=800,  # altura do gráfico
        margin=dict(l=0, r=0, t=0, b=0),  # reduz margens
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color="black"),
        hovermode=False,
        showlegend = True,
        legend=dict(
            x=0.9,
            y=0.7,
            font=dict(color='black')
        )
    )

    return fig