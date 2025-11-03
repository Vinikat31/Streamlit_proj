import pandas as pd
import itertools
import numpy as np
import streamlit as st
from scipy.stats import norm
import altair as alt

def extrair_tabela_marcas(df):
    """
    Extrai uma sub-tabela de um DataFrame com base nas marcas '#' (início) e '$' (fim).

    Funciona para tabelas com um ou múltiplos '$'.

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
    last_dollar = None

    # Procurar o primeiro '#' e o último '$'
    for row_idx, row in df.iterrows():
        for col_idx, value in row.items():
            if isinstance(value, str):
                if first_hash is None and '#' in value:
                    first_hash = (row_idx, col_idx)
                if '$' in value:
                    last_dollar = (row_idx, col_idx)  # sempre sobrescreve, pega o último

    if first_hash is None:
        raise ValueError("Nenhum '#' encontrado no DataFrame.")
    if last_dollar is None:
        raise ValueError("Nenhum '$' encontrado no DataFrame.")

    first_row, first_col = first_hash
    _, last_col = last_dollar  # não precisamos da linha do $

    # Selecionar colunas entre first_col e last_col
    cols = list(df.columns)
    first_col_idx = cols.index(first_col)
    last_col_idx = cols.index(last_col)
    if first_col_idx > last_col_idx:
        first_col_idx, last_col_idx = last_col_idx, first_col_idx
    selected_cols = cols[first_col_idx : last_col_idx + 1]

    # Pegar todas as linhas abaixo do cabeçalho até a última linha não vazia
    df_block = df.loc[first_row:, selected_cols]

    # Encontrar a primeira linha totalmente vazia para cortar (opcional)
    is_all_nan = df_block.isna().all(axis=1)
    if is_all_nan.any():
        last_data_idx = is_all_nan.idxmax() - 1
        df_block = df_block.loc[:last_data_idx]

    # Definir cabeçalho
    new_header = df_block.iloc[0]
    df_new_data = df_block[1:]
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
    print(col_resposta)
    if col_resposta not in df.columns:
        st.error(f"❌ Coluna de resposta '{col_resposta}' não encontrada.")
        return None, None
    
    y = pd.to_numeric(df[col_resposta], errors="coerce").fillna(0).values.reshape(-1, 1).astype(float)

    print(y)

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



def plot_efeito(df, df_desing):
    """
    Plota gráficos de porcentagem e probabilidade normal dos efeitos para todas as colunas de resposta
    que começam com '$', dentro de expanders no Streamlit.
    Inclui também a tabela de efeito e porcentagem em cada expander.
    Calcula automaticamente erro_efeito e t_val.
    """


    # Identifica todas as colunas de resposta que começam com '$'
    col_respostas = [col for col in df.columns if col.startswith("$")]

    for resposta in col_respostas:
        # Cria um expander para cada resposta
        with st.expander(f"📊 Gráficos para {resposta}", expanded=False):
            try:
                # Calcula efeito e porcentagem para a resposta específica
                efeito, porc = fabi_efeito(df[[resposta]], df_desing)

                # ------------------------
                # Calcula erro_efeito e t_val automaticamente
                # ------------------------
                erro_efeito = np.std(efeito)  # exemplo: desvio padrão dos efeitos
                t_val = 0.95  # exemplo: nível de confiança (pode vir de outra função)

                # ------------------------
                # Valida os dados
                # ------------------------
                if efeito is None or porc is None or len(efeito) == 0 or len(porc) == 0:
                    raise ValueError("Os dados de efeito ou porcentagem estão vazios.")
                if len(efeito) != len(porc):
                    raise ValueError("As listas 'efeito' e 'porc' devem ter o mesmo comprimento.")

                efeito = np.asarray(efeito, dtype=float)
                porc = np.asarray(porc, dtype=float)
                m = len(efeito)


                # ------------------------
                # Gráfico 1: Porcentagem dos efeitos
                # ------------------------
                df_graph1 = pd.DataFrame({
                    "Porcentagem (%)": np.round(porc, 2),
                    "Efeito_idx": np.arange(1, m + 1)
                })

                # ------------------------
                # Gráfico 2: Probabilidade normal
                # ------------------------
                D = np.argsort(efeito)
                C = efeito[D]

                A = np.zeros((m, 3))
                for i in range(1, m):
                    A[i, 0] = i / m
                for i in range(m):
                    A[i, 1] = (i + 1) / m
                for i in range(m):
                    A[i, 2] = (A[i, 0] + A[i, 1]) / 2

                B = norm.ppf(A[:, 2])
                B = np.asarray(B[:m], dtype=float)

                # Layout dos gráficos lado a lado
                col_graph1, col_graph2 = st.columns(2)

                # ---- Gráfico 1: Porcentagem ----
                with col_graph1:
                    chart1 = alt.Chart(df_graph1).mark_bar(color='mediumorchid').encode(
                        x=alt.X('Efeito_idx:O', title='Efeito'),
                        y=alt.Y('Porcentagem (%):Q', title='Porcentagem (%)')
                    ).properties(title='Porcentagem dos Efeitos')
                    linha0 = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 3]).encode(y='y:Q')
                    st.altair_chart(chart1 + linha0, use_container_width=True)

                # ---- Gráfico 2: Probabilidade Normal ----
                with col_graph2:
                    df_prob = pd.DataFrame({
                        'Efeito': C,
                        'Z': B,
                        'Label': (D + 1).astype(str)
                    })

                    base = alt.Chart(df_prob).encode(
                        x=alt.X('Efeito:Q', title='Efeito'),
                        y=alt.Y('Z:Q', title='Quantil Normal (Z)')
                    )

                    points = base.mark_point(shape='square', color='red', size=100)
                    text = base.mark_text(align='left', dx=5, dy=-5, color='black').encode(text='Label')
                    chart2 = (points + text).properties(title='Gráfico de Probabilidade Normal dos Efeitos')

                    # Linhas de erro
                    E = erro_efeito * t_val
                    linha_pos = alt.Chart(pd.DataFrame({'x': [E]})).mark_rule(color='red').encode(x='x:Q')
                    linha_neg = alt.Chart(pd.DataFrame({'x': [-E]})).mark_rule(color='red').encode(x='x:Q')

                    st.altair_chart(chart2 + linha_pos + linha_neg, use_container_width=True)

                # ------------------------
                # Cria tabela de efeito e porcentagem
                # ------------------------
                tabela_efeito = pd.DataFrame({
                    "Efeito": efeito,
                    "Porcentagem (%)": np.round(porc, 2)
                })
                st.markdown("### Tabela de Efeito e Porcentagem")
                st.dataframe(tabela_efeito)

            except Exception as e:
                # ------------------------
                # Mostra erro se houver problema nos dados
                # ------------------------
                col_graph1, col_graph2 = st.columns(2)
                for col in [col_graph1, col_graph2]:
                    with col:
                        st.altair_chart(
                            alt.Chart(pd.DataFrame({'x': [0], 'y': [0], 'erro': [f"❌ Erro: {e}"]}))
                            .mark_text(size=14, color='red', align='center', baseline='middle')
                            .encode(x='x', y='y', text='erro')
                            .properties(title='Erro ao gerar gráfico'),
                            use_container_width=True
                        )








