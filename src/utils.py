"""Funções auxiliares compartilhadas pelo projeto."""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("PassosMagicosAPI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Retorna o caminho raiz do projeto."""
    return PROJECT_ROOT


def load_json(path: str) -> dict:
    """Carrega um arquivo JSON e retorna o conteúdo como dicionário."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Salva um dicionário como arquivo JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def to_dataframe(obj: dict) -> pd.DataFrame:
    """Converte dict em DataFrame single-row, mantendo ordem por chaves."""
    if not obj:
        return pd.DataFrame()
    return pd.DataFrame([obj])


def setup_logging(log_dir: str = "logs", log_file: str = "api_monitor.log",
                  max_bytes: int = 10**6, backup_count: int = 5) -> logging.Logger:
    """Configura e retorna logger com RotatingFileHandler."""
    from logging.handlers import RotatingFileHandler

    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter("%(message)s")
    log_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    log_handler.setFormatter(log_formatter)

    _logger = logging.getLogger("PassosMagicosAPI")
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        _logger.addHandler(log_handler)

    return _logger


def log_event(event: str, status: str = "success", **extra) -> None:
    """Registra um evento estruturado no log."""
    entry = {
        "event": event,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        **extra,
    }
    logger.info(json.dumps(entry, ensure_ascii=False))
