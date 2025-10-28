import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
import altair as alt

def extrair_tabela_marcas(df):
    """
    Extrai uma sub-tabela de um DataFrame com base nas marcas '#' (início) e '@' (fim).

    A função:
      - Procura o primeiro '#' (define início)
      - Procura o último '$' (define fim)
      - Encontra o primeiro NaN abaixo do '#'
      - Seleciona as colunas entre '#' e '$'
      - Usa a primeira linha selecionada como cabeçalho da nova tabela

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    df_new : pd.DataFrame
        Novo DataFrame extraído entre as marcas.
    """

    first_hash = None
    last_at = None

    # Procurar o primeiro '#' e o último '@'
    for row_idx, row in df.iterrows():
        for col_idx, value in row.items():
            if isinstance(value, str):
                if first_hash is None and '#' in value:
                    first_hash = (row_idx, col_idx)
                if '$' in value:
                    last_at = (row_idx, col_idx)

    if first_hash is None:
        raise ValueError("Nenhum '#' encontrado no DataFrame.")
    if last_at is None:
        raise ValueError("Nenhum '$' encontrado no DataFrame.")

    # Posicionamentos
    first_row, first_col = first_hash
    last_row, last_col = last_at

    # Encontrar o primeiro NaN abaixo do primeiro '#'
    col_values = df[first_col]
    subsequent_values = col_values.iloc[first_row + 1:]
    nan_index = subsequent_values.index[subsequent_values.isna()].tolist()

    if not nan_index:
        end_row = df.index[-1]  # Última linha
    else:
        end_row = nan_index[0] - 1  # Linha antes do NaN

    # Selecionar colunas entre first_col e last_col
    cols = list(df.columns)
    first_col_idx = cols.index(first_col)
    last_col_idx = cols.index(last_col)
    if first_col_idx > last_col_idx:
        first_col_idx, last_col_idx = last_col_idx, first_col_idx

    selected_cols = cols[first_col_idx : last_col_idx + 1]

    # Fatiar DataFrame
    df_new = df.loc[first_row : end_row, selected_cols]

    # Definir cabeçalho
    new_header = df_new.iloc[0]
    df_new_data = df_new[1:]
    df_new_data.columns = new_header
    df_new = df_new_data.reset_index(drop=True)

    return df_new

def gera_design_fatorial(df):
    fatores = [col for col in df.columns if '#' in col]
    k = len(fatores)
    # pegar só as colunas dos fatores básicos, com valores -1 e 1
    X_base = df[fatores].copy()

    # copiar pra DataFrame novo que vamos expandir com interações
    df_design = pd.DataFrame()

    # fatores básicos
    for f in fatores:
        df_design[f] = X_base[f]

    # gerar interações de ordem 2 até k
    for i in range(2, k+1):
        for combo in itertools.combinations(fatores, i):
            col_name = ':'.join(combo)  # nome tipo "fator1:fator2"
            # produto das colunas dos fatores
            df_design[col_name] = X_base[list(combo)].prod(axis=1)

    # inserir coluna constante (intercepto) no início
    #df_design.insert(0, 'Intercepto', 1)

    return df_design





def fabi_efeito(df, df_desing):
    """
    Calcula os efeitos em planejamento fatorial.

    Retorna:
    --------
    efeito : np.ndarray
    porc : np.ndarray
    """
    # Converter X
    X = df_desing.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)

    # Coluna de resposta
    col_resposta = df.columns[-1]
    if not col_resposta:
        st.error("❌ Coluna de resposta '$resposta' não encontrada.")
        return None, None

    y = pd.to_numeric(df[col_resposta[0]], errors="coerce").fillna(0).values.reshape(-1, 1).astype(float)

    # Calcular efeitos usando pseudoinversa
    try:
        efeito = 2 * np.linalg.pinv(X.T @ X) @ (X.T @ y)
    except Exception as e:
        st.error(f"❌ Erro no cálculo dos efeitos: {e}")
        return None, None

    efeito = efeito.flatten()

    # Porcentagens
    soma_efeito2 = np.sum(efeito ** 2)
    porc = (efeito ** 2 / soma_efeito2) * 100
    porc = np.nan_to_num(porc)

    return efeito, porc



def plot_efeito(efeito, porc, erro_efeito=2, t=2):
    """
    Plota gráficos de porcentagem e probabilidade normal dos efeitos lado a lado no Streamlit.
    """

    m = len(efeito)

    # ------------------------
    # Criar coluna temporária para gráfico 1
    # ------------------------
    df_graph1 = pd.DataFrame({
        "Porcentagem (%)": np.round(porc, 2),
        "Efeito_idx": np.arange(1, m + 1)
    })

    # ------------------------
    # Ordenar efeitos para gráfico de probabilidade normal
    # ------------------------
    D = np.argsort(efeito)
    C = efeito[D].astype(float)

    # Percentis para gráfico normal
    A = np.zeros((m, 3))
    for i in range(1, m):
        A[i, 0] = i / m
    for i in range(m):
        A[i, 1] = (i + 1) / m
    for i in range(m):
        A[i, 2] = (A[i, 0] + A[i, 1]) / 2

    B = norm.ppf(A[:, 2])
    B = np.asarray(B[:m], dtype=float)

    # ------------------------
    # Layout: gráficos lado a lado
    # ------------------------
    col_graph1, col_graph2 = st.columns(2)

    # ---- Gráfico 1: Porcentagem dos efeitos ----
    with col_graph1:
        chart1 = alt.Chart(df_graph1).mark_bar(color='mediumorchid').encode(
            x='Efeito_idx:O',
            y='Porcentagem (%):Q'
        ).properties(title='Porcentagem dos Efeitos')
        st.altair_chart(chart1, use_container_width=True)

    # ---- Gráfico 2: Probabilidade normal dos efeitos ----
    with col_graph2:
        df_prob = pd.DataFrame({
            'Efeito': C,
            'Z': B,
            'Label': (D + 1).astype(str)
        })

        base = alt.Chart(df_prob).encode(
            x='Efeito:Q',
            y='Z:Q'
        )

        points = base.mark_point(shape='square', color='red', size=100)
        text = base.mark_text(
            align='left', dx=5, dy=-5, color='black'
        ).encode(text='Label')

        chart2 = (points + text).properties(title='Gráfico de Probabilidade Normal dos Efeitos')

        if erro_efeito != 0 and t != 0:
            E = erro_efeito * t

            # Linha positiva
            chart2 = chart2 + alt.Chart(pd.DataFrame({'x': [E], 'y0': [B.min()], 'y1': [B.max()]})).mark_rule(
                color='red'
            ).encode(
                x='x:Q',
                y='y0:Q',
                y2='y1:Q'
            )

            # Linha negativa
            chart2 = chart2 + alt.Chart(pd.DataFrame({'x': [-E], 'y0': [B.min()], 'y1': [B.max()]})).mark_rule(
                color='red'
            ).encode(
                x='x:Q',
                y='y0:Q',
                y2='y1:Q'
            )

        st.altair_chart(chart2, use_container_width=True)







