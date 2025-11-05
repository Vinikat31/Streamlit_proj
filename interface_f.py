import tempfile
import zipfile
import os
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np

# --- Funções para ZIP ---
def salvar_zip_temporario(uploaded_zip):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_zip.write(uploaded_zip.getbuffer())
    tmp_zip.flush()
    tmp_zip.close()
    return tmp_zip.name

def extrair_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

# --- Função para arquivos individuais ---
def salvar_arquivo_temporario(uploaded_file):
    """Salva arquivo individual (ex: .fid) em pasta temporária e retorna o caminho."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_dir

# --- Funções de processamento ---
def encontrar_pastas_com_pdata(folder_path):
    pastas_num = []
    for root, dirs, _ in os.walk(folder_path):
        if "pdata" in dirs:
            pastas_num.append(os.path.abspath(root))
    return pastas_num

def carregar_espectro_1D(data, udic, pasta, tipo):
    uc = ng.fileiobase.uc_from_udic(udic, dim=0)
    ppm = uc.ppm_scale()
    if np.all(data == 0):
        return None, f"⚠️ Espectro vazio: {pasta}"
    data_norm = data / np.max(np.abs(data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ppm, y=data_norm, mode='lines',
        name=f"{os.path.basename(pasta)} - {tipo}",
        line=dict(width=1, color='black')
    ))
    fig.update_layout(
        title=f"Espectro {tipo} - {os.path.basename(pasta)}",
        xaxis_title="Deslocamento químico (ppm)",
        yaxis_title="Intensidade (u.a.)",
        xaxis=dict(autorange='reversed'),
        template="plotly_white",
        height=800,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig, None

def carregar_espectro_2D(data, udic, pasta, tipo):
    uc_F1 = ng.fileiobase.uc_from_udic(udic, dim=0)
    uc_F2 = ng.fileiobase.uc_from_udic(udic, dim=1)
    ppm_F1 = uc_F1.ppm_scale()
    ppm_F2 = uc_F2.ppm_scale()
    data_log = np.log1p(np.abs(data))
    data_norm = (data_log - np.min(data_log)) / (np.max(data_log) - np.min(data_log))
    fig = go.Figure(data=go.Heatmap(
        z=data_norm, x=ppm_F2, y=ppm_F1,
        colorscale='Viridis', reversescale=True,
        colorbar=dict(title='Intensidade (log)')
    ))
    fig.update_layout(
        title=f"Espectro {tipo} - {os.path.basename(pasta)}",
        xaxis_title="F2 (ppm)",
        yaxis_title="F1 (ppm)",
        xaxis=dict(autorange='reversed'),
        yaxis=dict(autorange='reversed'),
        template="plotly_white",
        height=800,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def processar_pastas(pastas):
    tabs, figs, erros = [], [], []
    for pasta in pastas:
        pdata_dir = os.path.join(pasta, "pdata", "1")
        if not os.path.exists(pdata_dir):
            erros.append(f"❌ Pasta inválida ou sem 'pdata/1': {pdata_dir}")
            continue
        try:
            dic, data = ng.bruker.read_pdata(pdata_dir)
            tipo = dic.get('acqus', {}).get('PULPROG', 'Desconhecido')
            udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
            ndim = data.ndim

            if ndim == 1:
                fig, erro = carregar_espectro_1D(data, udic, pasta, tipo)
                if erro:
                    erros.append(erro)
                    continue
                figs.append(fig)
                tabs.append(f"{os.path.basename(pasta)} ({tipo})")
            elif ndim == 2:
                fig = carregar_espectro_2D(data, udic, pasta, tipo)
                figs.append(fig)
                tabs.append(f"{os.path.basename(pasta)} ({tipo})")
            else:
                erros.append(f"⚠️ Dimensão não suportada: {ndim}D em {pasta}")
        except Exception as e:
            erros.append(f"Erro ao carregar {pasta}: {e}")
    return tabs, figs, erros
