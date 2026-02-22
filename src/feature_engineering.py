import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas usadas no pipeline.

    - `avg_performance`: média de INDE/IDA/IPP/IEG quando disponíveis
    - `low_engagement`: 1 se avg_performance estiver abaixo da mediana
    """

    df = df.copy()

    base_cols = [c for c in ["INDE", "IDA", "IPP", "IEG"] if c in df.columns]
    if base_cols:
        df["avg_performance"] = df[base_cols].mean(axis=1)
        median_val = df["avg_performance"].median()
        df["low_engagement"] = (df["avg_performance"] < median_val).astype(int)

    return df


def select_features(df: pd.DataFrame, target: str = "risk_label") -> pd.DataFrame:
    """Remove colunas claramente de identificação, mantendo o alvo.

    - Remove colunas que parecem IDs ou identificadores sensíveis
    - Preserva a coluna `target` quando presente
    """

    df = df.copy()
    cols_to_drop = []

    for col in df.columns:
        if col == target:
            continue
        lower = col.lower()
        if any(x in lower for x in ["id", "nome", "cpf", "rg", "email", "fonte"]):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df