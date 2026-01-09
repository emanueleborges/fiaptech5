import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features a partir dos dados existentes."""
    # Exemplo: Criar feature de media de notas
    # df['media_geral'] = df[['nota1', 'nota2']].mean(axis=1)
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona as colunas que serao usadas no modelo."""
    return df
