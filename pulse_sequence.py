# spin_graph.py
from tabnanny import NannyNag

import plotly.graph_objects as go
import numpy as np


def inset_graph(theta, phi, omega_ref, wnut, phi_p, tau, time_relax):
    """
    Cria um gráfico secundário (inset) mostrando algum dado derivado
    do gráfico principal. Retorna um objeto Plotly Figure.

    Os parâmetros devem ser os mesmos do gráfico principal,
    para manter consistência com os inputs do usuário.
    """


    tempo = np.arange(0, 20, 0.01)

    # --- Linha de base ---
    d1 = np.zeros_like(tempo)
    d1[tempo >= 1] = np.nan
    # --- Pulso zg (como quadro) ---
    pulso = np.full_like(tempo, np.nan)
    pulso[(tempo >= 1) & (tempo <= 2)] = 1  # pulso de 1.5 a 2.5 ms

    # --- Sinal de aquisição (FID) ---
    fid = np.full_like(tempo, np.nan)  # NaN antes de começar
    idx_fid = tempo > 2.5  # FID começa após o pulso
    fid[idx_fid] = np.exp(-(tempo[idx_fid] - 2.5) / 3) * np.cos(2 * np.pi * 2 * (tempo[idx_fid] - 2.5))

    # --- Criar figura ---
    fig = go.Figure()

    # Linha de base
    fig.add_trace(go.Scatter(
        x=tempo, y=d1,
        mode='lines', line=dict(color='green', width=2),
        name='Linha de base'
    ))

    # Pulso como quadro preenchido
    fig.add_trace(go.Scatter(
        x=tempo, y=pulso,
        mode='lines',
        line=dict(color='blue', width=2),
        fill='tozeroy',  # preenche até zero
        name='Pulso ZG'
    ))

    # FID
    fig.add_trace(go.Scatter(
        x=tempo, y=fid,
        mode='lines',
        line=dict(color='red', width=2),
        name='FID'
    ))

    # --- Layout do inset ---
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=100,
        width=1000,
        legend=dict(
            x=0.9,
            y=0.7,
            font=dict(color='black')),
        annotations = [
            dict(
                showarrow=False,
                x=0.5,
                y=1,
                text='d1',
                font=dict(size=12, color="Green"),
                xanchor="center",  # Alinha o texto no centro no eixo X
                yanchor="bottom"  # Alinha o texto na parte inferior no eixo Y
            ),
            dict(
                showarrow=False,
                x=1.5,
                y=1,
                text='Pulse',
                font=dict(size=12, color="Blue"),
                xanchor="center",  # Alinha o texto no centro no eixo X
                yanchor="bottom"  # Alinha o texto na parte inferior no eixo Y
            )
        ]
    )

    return fig
