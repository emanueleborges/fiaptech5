import pandas as pd
from typing import Union
import pdfplumber


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega os dados de CSV ou Excel e retorna um DataFrame.

    Suporta arquivos com extensão .csv, .xls, .xlsx.
    """
    try:
        if filepath.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            df = load_csv_with_fallback(filepath)
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


def load_csv_with_fallback(filepath: str) -> pd.DataFrame:
    """
    Tenta carregar um arquivo CSV com diferentes configurações de separador e codificação.
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    separators = [',', ';', '\t']

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                if not df.empty:
                    return df
            except Exception:
                continue

    raise ValueError(f"Não foi possível carregar o arquivo CSV: {filepath}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza inicial:
    - Remove duplicatas
    - Normaliza nomes de colunas (strip, lower)
    - Não remove linhas com nulos (imputação fica para o pipeline)
    """
    if df is None or df.empty or df.columns.size == 0:
        return pd.DataFrame()

    # Normalizar nomes de colunas
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Remover duplicatas estritas
    df = df.drop_duplicates()

    return df


def sample_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Gera um dataset sintético baseado nos indicadores da Passos Mágicos.
    
    Indicadores:
    - INDE: Índice do Desenvolvimento Educacional
    - IAA: Indicador de Autoavaliação
    - IEG: Indicador de Engajamento
    - IPS: Indicador de Psicossocial
    - IPP: Indicador de Psicopedagógico
    - IPV: Indicador de Ponto de Virada
    - IDA: Indicador de Aprendizagem
    """
    import numpy as np

    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        'INDE': rng.normal(7.0, 1.5, size=n).clip(0, 10),
        'IAA': rng.normal(8.0, 1.0, size=n).clip(0, 10),
        'IEG': rng.normal(7.5, 1.2, size=n).clip(0, 10),
        'IPS': rng.normal(7.0, 1.3, size=n).clip(0, 10),
        'IPP': rng.normal(7.2, 1.4, size=n).clip(0, 10),
        'IDA': rng.normal(6.5, 1.8, size=n).clip(0, 10),
        'FASE': rng.randint(0, 8, size=n),
        'PEDRA': rng.choice(['Ametista', 'Topázio', 'Brilhante', 'Quartzo'], size=n),
    })
    
    # Target: 1 se INDE < 6.0 ou IDA < 5.5 (Risco de defasagem)
    df['risk_label'] = ((df['INDE'] < 6.0) | (df['IDA'] < 5.5)).astype(int)
    
    # Adicionar alguns nulos para testar o imputer
    for col in ['INDE', 'IAA', 'FASE']:
        df.loc[rng.choice(df.index, size=int(n*0.05)), col] = np.nan
        
    return df


def prepare_real_data(df: Union[str, pd.DataFrame]) -> tuple:
    """
    Prepara os dados reais para o treinamento:
    - Realiza limpeza inicial e define target/features.
    - Recebe um DataFrame já carregado ou um filepath para carregar.
    
    Args:
        df: DataFrame carregado ou caminho do arquivo para carregar
        
    Returns:
        tuple: (features DataFrame, target DataFrame)
    """
    if isinstance(df, str):
        # Se receber um filepath, carrega o arquivo
        df = load_data(df)
    
    df = clean_data(df)

    # Definir target e features
    df['risk_label'] = ((df['INDE'] < 6.0) | (df['IDA'] < 5.5)).astype(int)
    features = df.drop(columns=['risk_label'])
    target = df[['risk_label']]

    return features, target


def extract_data_from_pdf(filepath: str) -> pd.DataFrame:
    """
    Extrai dados de um arquivo PDF e retorna um DataFrame.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            data = []
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    data.extend(table)

        # Converter para DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])  # Usar a primeira linha como cabeçalho
        return df
    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")
        return pd.DataFrame()
