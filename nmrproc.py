import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import nmrglue as ng
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Para AsLS e Whittaker
from scipy import sparse
from scipy.sparse.linalg import spsolve

# tentamos importar simpson; se não existir, usamos trapz
try:
    from scipy.integrate import simpson
    _HAS_SIMPSON = True
except Exception:
    _HAS_SIMPSON = False
    from numpy import trapz as _trapz


def read_bruker(folder_path, calib_range=(-0.05, 0.05), tol_ppm=0.1):
    """
    Lê espectros Bruker processados de múltiplas pastas, calibra e integra todos em um único DataFrame.
    Ao final, o DataFrame é transposto (amostras nas linhas, ppm nas colunas).

    Parâmetros
    ----------
    folder_path : str
        Caminho da pasta principal que contém as subpastas com os espectros (ex: 'C:\\Users\\vinic\\Documents\\IC-RMN\\Parte1')
    calib_range : tuple (float, float)
        Faixa em ppm usada para calibrar o eixo (padrão: -0.05 a 0.05).
    tol_ppm : float
        Tolerância em ppm para o merge aproximado (padrão: 0.1).

    Retorna
    -------
    df_nmr : pd.DataFrame
        DataFrame com amostras nas linhas e valores de ppm nas colunas.
    pastas_validas : list
        Lista com os nomes das pastas que foram lidas com sucesso.
    """

    items = os.listdir(folder_path)
    pastas = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    df_nmr = None
    pastas_validas = []

    for pasta in pastas:
        caminho = os.path.join(folder_path, pasta, r'2\pdata\1')
        fid_path = os.path.join(folder_path, pasta, r'2\fid')

        # Verifica se os arquivos necessários existem
        if not os.path.exists(caminho) or not os.path.isfile(fid_path):
            print(f"❌ Pasta ignorada: {pasta} (sem dados válidos)")
            continue

        try:
            # Lê os dados processados
            dic, data = ng.bruker.read_pdata(caminho)

            # Cria dicionário universal e escala ppm
            udic = ng.bruker.guess_udic(dic, data, strip_fake=True)
            uc = ng.fileiobase.uc_from_udic(udic, dim=0)
            ppm = uc.ppm_scale()

            # Cria DataFrame temporário
            df_add = pd.DataFrame({'ppm': ppm, f'{pasta}': data})

            if df_nmr is None:
                # Primeiro espectro define o eixo base
                df_nmr = df_add.sort_values(by='ppm').reset_index(drop=True)
            else:
                # Calibração automática
                ppm_calibrate = ppm[(ppm > calib_range[0]) & (ppm < calib_range[1])]
                data_calibrate = data[(ppm > calib_range[0]) & (ppm < calib_range[1])]

                if len(ppm_calibrate) > 0:
                    idx_max = np.argmax(data_calibrate)
                    ppm_max = ppm_calibrate[idx_max]
                    df_add['ppm'] = ppm - ppm_max

                df_add = df_add.sort_values(by='ppm').reset_index(drop=True)
                # Junção aproximada
                df_nmr = pd.merge_asof(df_nmr, df_add, on='ppm', tolerance=tol_ppm, direction='nearest')

            pastas_validas.append(pasta)

        except Exception as e:
            print(f"⚠️ Erro ao ler {pasta}: {e}")

    if df_nmr is not None:
        print(f"✅ Integração concluída. Dimensões finais (antes da transposição): {df_nmr.shape}")

        # --- 1. Transpõe o DataFrame ---
        df_nmr = df_nmr.T

        # --- 2. Define a primeira linha como nomes das colunas ---
        df_nmr.columns = df_nmr.iloc[0]
        df_nmr = df_nmr.drop(df_nmr.index[0])

        # Converte valores para float
        df_nmr = df_nmr.astype(float)

        print(f"✅ DataFrame final: {df_nmr.shape} (amostras x ppm)")
    else:
        print("⚠️ Nenhum espectro foi lido com sucesso.")

    return df_nmr, pastas_validas

import os

def encontrar_pdata(root_dir):
    """
    Procura recursivamente todas as pastas 'pdata' dentro de root_dir.
    Retorna um dicionário {nome_da_subpasta: caminho_para_pdata}.
    """
    pdata_dict = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "pdata" in dirnames:
            pasta_nome = os.path.basename(dirpath)
            pdata_dict[pasta_nome] = os.path.join(dirpath, "pdata")
    return pdata_dict



def norma_nmr(df_nmr, faixa_ppm=(-0.5, 10), metodo='simpson', plotar=True):
    """
    Normaliza espectros de RMN com base na integral de uma faixa específica de ppm,
    mas aplica a normalização a todo o espectro.

    Parâmetros
    ----------
    df_nmr : pd.DataFrame
        DataFrame com amostras nas linhas e ppm nas colunas.
    faixa_ppm : tuple (float, float)
        Faixa de ppm usada apenas para o cálculo da integral.
    metodo : str
        Método de integração: 'simpson' (mais preciso) ou 'soma' (simples).
    plotar : bool
        Se True, plota gráficos comparativos antes e depois da normalização.

    Retorna
    -------
    df_normalizado : pd.DataFrame
        Espectros normalizados pelo valor integral da faixa especificada.
    integrais : pd.Series
        Integrais calculadas para cada amostra.
    """

    # --- 1. Separa valores e metadados ---
    df_valores = df_nmr.select_dtypes(include='number').copy()
    ppm_numeric = df_valores.columns.astype(float)
    df_metadata = df_nmr.select_dtypes(exclude='number')

    # --- 2. Seleciona apenas faixa de integração ---
    mascara_faixa = (ppm_numeric > faixa_ppm[0]) & (ppm_numeric < faixa_ppm[1])
    ppm_faixa = ppm_numeric[mascara_faixa]
    df_faixa = df_valores.loc[:, mascara_faixa]

    # --- 3. Calcula espaçamento médio ---
    dx = np.mean(np.diff(ppm_faixa))

    # --- 4. Calcula integrais na faixa ---
    if metodo == 'simpson':
        integrais = df_faixa.apply(lambda y: simpson(y, x=ppm_faixa), axis=1)
    elif metodo == 'soma':
        integrais = df_faixa.apply(lambda y: np.sum(y) * abs(dx), axis=1)
    else:
        raise ValueError("Método inválido. Use 'simpson' ou 'soma'.")

    # --- 5. Normaliza todo o espectro usando essas integrais ---
    df_normalizado = df_valores.div(integrais, axis=0)
    df_normalizado_completo = pd.concat([df_metadata, df_normalizado], axis=1)

    # --- 6. (Opcional) Plot comparativo ---
    if plotar:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for amostra in df_valores.index:
            axes[0].plot(ppm_numeric, df_valores.loc[amostra], lw=1)
            axes[1].plot(ppm_numeric, df_normalizado.loc[amostra], lw=1)

        axes[0].invert_xaxis()
        axes[0].set_title("Antes da normalização")
        axes[0].set_xlabel("Deslocamento químico (ppm)")
        axes[0].set_ylabel("Intensidade (u.a.)")

        axes[1].invert_xaxis()
        axes[1].set_title("Depois da normalização")
        axes[1].set_xlabel("Deslocamento químico (ppm)")
        axes[1].set_ylabel("Intensidade (u.a.)")

        plt.tight_layout()
        plt.show()

    print("✅ Normalização concluída com base na faixa de integração:", faixa_ppm)
    return df_normalizado_completo, integrais


def _area_by_method(x, y, method='integral'):
    """Retorna a área (ou métrica) de y(x) conforme method."""
    if len(x) < 1:
        return 0.0
    if method == 'integral':
        if len(x) < 2:
            return 0.0
        if _HAS_SIMPSON:
            return float(simpson(y, x))
        else:
            return float(_trapz(y, x))
    elif method == 'sum':
        # soma das intensidades vezes dx médio (Riemann aproximado)
        dx = np.mean(np.diff(x)) if len(x) > 1 else 0.0
        return float(np.sum(y) * abs(dx))
    elif method == 'mean':
        return float(np.mean(y)) if len(y) > 0 else 0.0
    else:
        raise ValueError("method deve ser 'sum', 'mean' ou 'integral'")


def baseline_c(df, metodo='polinomial', grau=3, window_length=11, polyorder=3, lam=1e5, p=0.01, plotar=False):
    """
    Corrige a linha de base (baseline) de espectros usando diferentes métodos.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame onde cada linha é uma amostra e cada coluna é um valor de ppm (float).
    metodo : str
        Método de correção: 'polinomial', 'piecewise', 'sgolay', 'asls', 'whittaker'
    grau : int
        Grau do polinômio (usado apenas se metodo='polinomial')
    window_length : int
        Janela do filtro Savitzky-Golay (usado se metodo='sgolay')
    polyorder : int
        Grau do filtro Savitzky-Golay (usado se metodo='sgolay')
    lam : float
        Parâmetro de suavidade para AsLS/Whittaker
    p : float
        Assimetria para AsLS
    plotar : bool
        Se True, plota espectro original e corrigido

    Retorna
    -------
    df_corrigido : pd.DataFrame
        DataFrame com espectros corrigidos
    """
    ppm = df.columns.astype(float)
    df_corrigido = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    def asls(y, lam=1e5, p=0.01, niter=10):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.diags(w, 0)
            Z = W + lam * D.T @ D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def whittaker_smooth(y, lam=1e5):
        L = len(y)
        E = sparse.eye(L)
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
        Z = E + lam * D.T @ D
        z = spsolve(Z, y)
        return z

    for amostra in df.index:
        y = df.loc[amostra].values.astype(float)

        if metodo == 'polinomial':
            coeffs = np.polyfit(ppm, y, grau)
            baseline = np.polyval(coeffs, ppm)

        elif metodo == 'piecewise':
            # pontos mínimos a cada 5% do espectro
            nseg = max(int(len(ppm) / 20), 2)
            idx = np.linspace(0, len(ppm) - 1, nseg, dtype=int)
            minima = y[idx]
            f = interp1d(ppm[idx], minima, kind='linear', fill_value='extrapolate')
            baseline = f(ppm)

        elif metodo == 'sgolay':
            baseline = savgol_filter(y, window_length=window_length, polyorder=polyorder)

        elif metodo == 'asls':
            baseline = asls(y, lam=lam, p=p)

        elif metodo == 'whittaker':
            baseline = whittaker_smooth(y, lam=lam)

        else:
            raise ValueError("Método inválido. Use 'polinomial', 'piecewise', 'sgolay', 'asls', ou 'whittaker'.")

        df_corrigido.loc[amostra] = y - baseline

    if plotar:
        import matplotlib.pyplot as plt
        for amostra in df.index[:3]:  # plota só as primeiras 3 amostras
            plt.figure(figsize=(10, 4))
            plt.plot(ppm, df.loc[amostra], label='Original')
            plt.plot(ppm, df_corrigido.loc[amostra], label='Corrigido')
            plt.gca().invert_xaxis()
            plt.title(f"Amostra {amostra}")
            plt.xlabel('ppm')
            plt.ylabel('Intensidade')
            plt.legend()
            plt.show()

    return df_corrigido


def plot_plotly(df, n=10, titulo="Espectros de RMN (Plotly)"):
    """
    Plota n espectros de RMN em um gráfico interativo do Plotly.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com amostras nas linhas e ppm nas colunas.
    n : int
        Número de espectros a serem plotados.
    titulo : str
        Título do gráfico.
    """

    ppm = df.columns.astype(float)
    fig = go.Figure()

    # Garantir que não passe do número total de amostras
    n = min(n, len(df))

    for amostra in df.index[:n]:
        y = df.loc[amostra].astype(float)
        fig.add_trace(go.Scatter(
            x=ppm,
            y=y,
            mode='lines',
            name=str(amostra)
        ))

    # Layout padrão de RMN
    fig.update_layout(
        title=titulo,
        xaxis_title='ppm',
        yaxis_title='Intensidade',
        xaxis=dict(autorange='reversed'),  # inverte eixo ppm
        template='simple_white',
        width=900,
        height=500,
        legend_title='Amostras'
    )

    fig.show()

def nmr_binning(ppm, intensidades, n_bins=None, bin_width=None,
                method='integral', return_bin_edges=False):
    """
    Faz binning de espectros NMR e calcula valores por bin.

    Parâmetros
    ----------
    ppm : array-like
        Eixo ppm (1D), em ordem crescente ou decrescente (tratado internamente).
    intensidades : array-like, pd.Series ou pd.DataFrame
        - Se 1D/Series: valores de uma amostra.
        - Se DataFrame: cada linha é uma amostra e as colunas devem corresponder aos pontos de `ppm`
          (ou ser indexadas na mesma ordem dos pontos).
    n_bins : int, opcional
        Número de bins desejados. Usado se bin_width for None. Default = 100.
    bin_width : float, opcional
        Largura do bin em ppm. Se fornecido, tem prioridade sobre n_bins.
    method : {'sum','mean','integral'}
        Como agregar os pontos dentro de cada bin:
         - 'sum' -> soma * dx médio (aproximação de área)
         - 'integral' -> simpson (ou trapz se simpson não disponível)
         - 'mean' -> média das intensidades no bin (integral total será média * largura do bin)
    return_bin_edges : bool
        Se True, retorna também os edges (bins) como terceiro valor.

    Retorno
    -------
    df_bins : pd.DataFrame
        DataFrame com bins (center ppm) como colunas e amostras como linhas.
    integrals : pd.Series
        Integral total de cada amostra (área), calculada a partir do método escolhido.
        - Para 'mean' a integral total = sum(bin_mean * bin_width)
        - Para 'sum' e 'integral' a integral total = soma das áreas dos bins
    (opcional) bin_edges : np.ndarray
        Os edges dos bins (len = n_bins+1)
    """
    ppm = np.asarray(ppm, dtype=float)
    # se ppm estiver decrescente, inverter para simplificar
    reversed_axis = False
    if ppm[0] > ppm[-1]:
        ppm = ppm[::-1]
        reversed_axis = True

    # normalizar intensidades para DataFrame (linhas = amostras)
    if isinstance(intensidades, pd.DataFrame):
        df_in = intensidades.copy()
        # se as colunas forem strings que representam ppm, tentar converter para float para alinhar
        try:
            cols_float = df_in.columns.astype(float)
            df_in.columns = cols_float
        except Exception:
            pass
        # garantir que as colunas correspondam ao eixo ppm em ordem
        # se colunas forem iguais ao ppm, reindex na ordem de ppm
        if set(df_in.columns) == set(ppm):
            df_in = df_in.reindex(columns=ppm)  # ordena pelas posições do ppm array
        else:
            # se shape casar por posição, assumimos mesma ordem: (WARN) se não, usuário deve fornecer DataFrame alinhado
            if df_in.shape[1] != len(ppm):
                raise ValueError("DataFrame de intensidades não corresponde ao eixo ppm (n cols != len(ppm)).")
    else:
        # Series ou 1D array -> transformar em DataFrame com uma linha
        arr = np.asarray(intensidades)
        if arr.ndim == 1:
            df_in = pd.DataFrame([arr], index=['sample_0'])
        else:
            raise ValueError("intensidades deve ser 1D (Series/array) ou DataFrame")

    # definir bins
    if bin_width is not None:
        if bin_width <= 0:
            raise ValueError("bin_width deve ser > 0")
        # edges cobrindo o min/max de ppm
        n_bins_calc = int(np.ceil((ppm.max() - ppm.min()) / bin_width))
        bin_edges = np.linspace(ppm.min(), ppm.max(), n_bins_calc + 1)
    else:
        if n_bins is None:
            n_bins = 100
        if n_bins <= 0:
            raise ValueError("n_bins deve ser > 0")
        bin_edges = np.linspace(ppm.min(), ppm.max(), n_bins + 1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins_eff = len(bin_centers)

    # preparar DataFrame de saída
    df_bins = pd.DataFrame(0.0, index=df_in.index, columns=bin_centers)

    # para cada bin, calcular máscara e agregar
    # usamos indices por posição (digitize)
    bin_indices = np.digitize(ppm, bin_edges) - 1  # valores 0..n_bins_eff-1 (ou -1 / n_bins se fora)

    for i in range(n_bins_eff):
        mask = bin_indices == i
        if not mask.any():
            # bin vazio -> já está zero
            continue

        x = ppm[mask]
        width = bin_edges[i+1] - bin_edges[i]
        for amostra in df_in.index:
            y = df_in.loc[amostra].values[mask]
            if method == 'mean':
                val = _area_by_method(x, y, method='mean')  # mean
                df_bins.at[amostra, bin_centers[i]] = val
            elif method == 'sum':
                # soma * dx médio (área aproximada)
                val = _area_by_method(x, y, method='sum')
                df_bins.at[amostra, bin_centers[i]] = val
            elif method == 'integral':
                val = _area_by_method(x, y, method='integral')
                df_bins.at[amostra, bin_centers[i]] = val
            else:
                raise ValueError("method inválido: use 'sum','mean' ou 'integral'")


    # se o eixo original estava decrescente, opcionalmente inverter as colunas para manter ordem decrescente:
    if reversed_axis:
        df_bins = df_bins[df_bins.columns[::-1]]

    if return_bin_edges:
        return df_bins, bin_edges
    return df_bins




