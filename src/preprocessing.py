import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(filepath: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV/Excel."""
    try:
        # Tenta carregar CSV por padrao, ajustar conforme necessidade
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza a limpeza inicial dos dados (tratamento de nulos, duplicatas)."""
    # Exemplo: df = df.dropna()
    return df
