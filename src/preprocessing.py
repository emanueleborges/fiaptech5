import os
from pathlib import Path

import numpy as np
import pandas as pd


def _normalizar_nome_coluna(col: str) -> str:
    """Normaliza nomes de colunas para um formato padronizado.

    - Remove espaços extras
    - Mapeia colunas conhecidas (INDE 22 -> INDE, Pedra 22 -> PEDRA, etc.)
    - Converte demais para snake_case minúsculo
    """

    if not isinstance(col, str):
        col = str(col)

    original = col
    col = col.strip()
    upper = col.upper()

    if upper.startswith("INDE"):
        base = "INDE"
    elif "PEDRA" in upper:
        base = "PEDRA"
    elif upper.startswith("IDA"):
        base = "IDA"
    elif upper.startswith("IEG"):
        base = "IEG"
    elif upper.startswith("IAA"):
        base = "IAA"
    elif upper.startswith("IPS"):
        base = "IPS"
    elif upper.startswith("IPP"):
        base = "IPP"
    elif upper.startswith("IPV"):
        base = "IPV"
    elif upper.startswith("IAN"):
        base = "IAN"
    elif "DESTAQUE IPV" in upper:
        base = "destaque_ipv"
    else:
        base = original.strip().lower()
        base = base.replace(" ", "_")
    return base


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e normaliza colunas básicas.

    - Normaliza nomes de colunas (INDE 22 -> INDE, Pedra 22 -> PEDRA, etc.)
    - Garante colunas minúsculas para casos genéricos (" A " -> "a")
    - Desduplica nomes repetidos (ex.: Destaque IPV -> destaque_ipv_1, destaque_ipv_2)
    - Remove linhas totalmente duplicadas.
    """

    col_map = {}
    counts = {}
    for col in df.columns:
        base = _normalizar_nome_coluna(col)
        # Se for genérico (não mapeado específico), garantir minúsculo
        if base not in {"INDE", "PEDRA", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "IAN", "destaque_ipv"}:
            base = base.lower()

        counts.setdefault(base, 0)
        counts[base] += 1

        if counts[base] == 1:
            new_name = base
        else:
            new_name = f"{base}_{counts[base]}"

        col_map[col] = new_name

    df = df.copy()
    df.columns = [col_map[c] for c in df.columns]

    # Remover linhas duplicadas
    df = df.drop_duplicates()

    return df


def _coerce_comma_float(series: pd.Series) -> pd.Series:
    """Converte strings com vírgula decimal para float."""

    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def sample_data(n: int = 100) -> pd.DataFrame:
    """Gera um dataset sintético com colunas principais e `risk_label`.

    Usado tanto em testes quanto como fallback quando não há dados reais.
    """

    rng = np.random.default_rng(42)

    inde = rng.normal(loc=7.0, scale=1.0, size=n)
    ida = rng.normal(loc=7.0, scale=1.0, size=n)
    ieg = rng.normal(loc=7.0, scale=1.0, size=n)
    iaa = rng.normal(loc=7.0, scale=1.0, size=n)
    ips = rng.normal(loc=7.0, scale=1.0, size=n)
    ipp = rng.normal(loc=7.0, scale=1.0, size=n)
    fase = rng.integers(1, 4, size=n)
    pedras = np.array(["Ametista", "Quartzo", "Topazio"])

    df = pd.DataFrame(
        {
            "INDE": inde,
            "IDA": ida,
            "IEG": ieg,
            "IAA": iaa,
            "IPS": ips,
            "IPP": ipp,
            "FASE": fase,
            "PEDRA": pedras[rng.integers(0, len(pedras), size=n)],
        }
    )

    # Definir risco como 25% piores em INDE
    threshold = np.quantile(df["INDE"], 0.25)
    df["risk_label"] = (df["INDE"] < threshold).astype(int)

    return df


def _is_pdf_file(path: str) -> bool:
    """Detecta assinatura de arquivo PDF pelos primeiros bytes."""

    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header.startswith(b"%PDF")
    except OSError:
        return False


def _is_zip_file(path: str) -> bool:
    """Detecta assinatura de arquivo ZIP pelos primeiros bytes."""

    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header.startswith(b"PK\x03\x04")
    except OSError:
        return False


def load_data(path: str) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV/Excel simples.

    Nos testes é usado apenas com CSV; aqui mantemos suporte básico.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    if path_obj.suffix.lower() == ".csv":
        # Tenta UTF-8 primeiro; se falhar, usa encodings comuns em Windows/português
        encodings_to_try = ["utf-8", "latin-1", "cp1252"]
        last_error = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(path_obj, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
        else:
            # Se nenhum encoding funcionou, relança o último erro
            raise last_error
    elif path_obj.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path_obj)
    else:
        # Fallback genérico
        df = pd.read_csv(path_obj)

    return df


def prepare_real_data(df: pd.DataFrame):
    """Prepara dados reais já carregados em um DataFrame.

    - Converte colunas de indicadores (INDE, IDA, etc.) com vírgula decimal para float
    - Gera coluna alvo binária `risk_label` com base em INDE
    """

    # Normalizar colunas primeiro
    df = clean_data(df)

    required = {"INDE", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "IAN"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes para dados reais: {missing}")

    num_cols = list(required)
    for col in num_cols:
        df[col] = _coerce_comma_float(df[col])

    # Criar target baseado em INDE (pior INDE = maior risco)
    inde = df["INDE"].copy()
    threshold = inde.mean()
    risk = (inde < threshold).astype(int)

    X = df[num_cols].copy()
    y = risk

    return X, y