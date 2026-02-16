import pandas as pd
from typing import List


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features baseadas nos indicadores educacionais.
    """
    df = df.copy()

    # Média de indicadores de desempenho
    perf_cols = ['INDE', 'IDA', 'IPP']
    if all(col in df.columns for col in perf_cols):
        df['avg_performance'] = df[perf_cols].mean(axis=1)

    # Flag de baixo engajamento
    if 'IEG' in df.columns:
        df['low_engagement'] = (df['IEG'] < 5.0).astype(int)

    return df


def select_features(df: pd.DataFrame, target: str = 'risk_label') -> pd.DataFrame:
    """Retorna DataFrame apenas com features e a coluna target (se existir).

    A seleção aqui é simples: todas as colunas numéricas + categóricas
    (serão tratadas no pipeline) exceto identificadores óbvios.
    """
    df = df.copy()
    # Excluir colunas que claramente não são features
    drop_like = [c for c in df.columns if c.lower() in ('id', 'identificador')]
    df = df.drop(columns=drop_like, errors='ignore')

    # Garantir que target, se existir, venha por último
    if target in df.columns:
        cols = [c for c in df.columns if c != target] + [target]
        df = df[cols]

    return df
