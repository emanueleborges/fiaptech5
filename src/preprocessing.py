import pandas as pd
from typing import Union
import pdfplumber
import re
import unicodedata


def _is_pdf_file(filepath: str) -> bool:
    """Detecta PDF pela assinatura binária (%PDF), independente da extensão."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
        return header == b"%PDF"
    except Exception:
        return False


def _is_zip_file(filepath: str) -> bool:
    """Detecta ZIP (ex.: docx/xlsx) pela assinatura binária PK."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(2)
        return header == b"PK"
    except Exception:
        return False


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega os dados de CSV ou Excel e retorna um DataFrame.

    Suporta arquivos com extensão .csv, .xls, .xlsx.
    """
    try:
        if _is_zip_file(filepath) and not filepath.lower().endswith((".xls", ".xlsx")):
            raise ValueError("Arquivo parece ser ZIP/DOCX e nao um CSV/Excel valido.")

        if _is_pdf_file(filepath):
            print("Arquivo detectado como PDF por assinatura binária. Extraindo tabelas...")
            return extract_data_from_pdf(filepath)

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

    def _normalize_col(col: str) -> str:
        col = str(col).strip().lower()
        col = unicodedata.normalize("NFKD", col)
        col = "".join(ch for ch in col if not unicodedata.combining(ch))
        col = re.sub(r"\s+", "_", col)
        col = re.sub(r"[^a-z0-9_]+", "", col)
        return col

    def _make_unique(cols):
        seen = {}
        unique = []
        for col in cols:
            count = seen.get(col, 0)
            if count == 0:
                unique.append(col)
            else:
                unique.append(f"{col}_{count + 1}")
            seen[col] = count + 1
        return unique

    # Normalizar nomes de colunas
    df = df.copy()
    normalized_cols = [_normalize_col(c) for c in df.columns]
    df.columns = _make_unique(normalized_cols)

    # Mapear colunas do PEDE para nomes esperados pelo pipeline
    col_map = {
        "fase": "FASE",
        "turma": "TURMA",
        "pedra_20": "PEDRA",
        "pedra_21": "PEDRA",
        "pedra_22": "PEDRA",
        "pedra_23": "PEDRA",
        "pedra_24": "PEDRA",
        "inde_22": "INDE",
        "inde_23": "INDE",
        "inde_24": "INDE",
        "inde_2022": "INDE",
        "inde_2023": "INDE",
        "inde_2024": "INDE",
        "ida": "IDA",
        "ieg": "IEG",
        "iaa": "IAA",
        "ips": "IPS",
        "ipp": "IPP",
        "ipv": "IPV",
        "ian": "IAN",
    }

    rename_cols = {c: col_map[c] for c in df.columns if c in col_map}
    if rename_cols:
        df = df.rename(columns=rename_cols)

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


def prepare_real_data(data_source: Union[str, pd.DataFrame]):
    """
    Prepara os dados reais para o treinamento:
    - Carrega os dados do arquivo especificado (ou usa DataFrame já carregado).
    - Realiza limpeza inicial e define target/features.
    """
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        df = load_data(data_source)

    df = clean_data(df)

    if df.empty:
        raise ValueError("Dataset vazio após carregamento/limpeza.")

    required_cols = ["INDE", "IDA"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes para criar target: {missing_cols}")

    def _select_latest(df_in: pd.DataFrame, base: str, year_candidates):
        if base in df_in.columns:
            return base
        for cand in year_candidates:
            if cand in df_in.columns:
                return cand
        return None

    # Usar coluna mais recente para INDE/PEDRA se necessario
    inde_col = _select_latest(df, "INDE", ["INDE_24", "INDE_2024", "INDE_23", "INDE_2023", "INDE_22", "INDE_2022"])
    if inde_col and inde_col != "INDE":
        df["INDE"] = df[inde_col]

    pedra_col = _select_latest(df, "PEDRA", ["PEDRA_24", "PEDRA_23", "PEDRA_22", "PEDRA_21", "PEDRA_20"])
    if pedra_col and pedra_col != "PEDRA":
        df["PEDRA"] = df[pedra_col]

    # Coagir colunas numericas com virgula
    numeric_like = [
        "INDE",
        "IDA",
        "IEG",
        "IAA",
        "IPS",
        "IPP",
        "IPV",
        "IAN",
    ]
    for col in numeric_like:
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Definir target e features
    if 'risk_label' not in df.columns:
        df['risk_label'] = ((df['INDE'] < 6.0) | (df['IDA'] < 5.5)).astype(int)
    target = df['risk_label']
    features = df.drop(columns=['risk_label'])

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

        if not data:
            return pd.DataFrame()

        # Primeira linha como cabecalho; ajustar linhas com tamanho diferente
        header = [str(h).strip() for h in data[0]]
        header_len = len(header)
        cleaned_rows = []
        for row in data[1:]:
            if row is None:
                continue
            row = list(row)
            if len(row) < header_len:
                row = row + [None] * (header_len - len(row))
            elif len(row) > header_len:
                row = row[:header_len]
            cleaned_rows.append(row)

        df = pd.DataFrame(cleaned_rows, columns=header)
        return df
    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")
        return pd.DataFrame()
