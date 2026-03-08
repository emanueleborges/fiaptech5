"""Módulo de monitoramento e logging estruturado para a API Passos Mágicos.

Responsável por:
- Logging estruturado em JSON (RotatingFileHandler)
- Métricas operacionais (requests, latência, erros, uptime)
- Detecção de drift em tempo real e em batch
- Histórico de predições para análise
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────
# 1. Logger estruturado em JSON
# ──────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Formatter que serializa cada registro de log como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Mesclar campos extras anexados ao record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logger(
    name: str = "PassosMagicosAPI",
    log_dir: str = "logs",
    log_file: str = "api_monitor.log",
    max_bytes: int = 5 * 10**6,
    backup_count: int = 5,
    level: int = logging.INFO,
) -> logging.Logger:
    """Cria e configura o logger principal com RotatingFileHandler + JSON.

    Também adiciona um StreamHandler para saída no console.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicação de handlers em reloads
    if logger.handlers:
        return logger

    json_formatter = JSONFormatter()

    # Handler: arquivo rotativo
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Handler: console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: str = "info",
    **kwargs,
) -> None:
    """Registra um evento estruturado com campos extras."""
    record = logger.makeRecord(
        name=logger.name,
        level=getattr(logging, level.upper(), logging.INFO),
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.extra_fields = {"event": event, **kwargs}
    logger.handle(record)


# ──────────────────────────────────────────────
# 2. Métricas operacionais (in-memory)
# ──────────────────────────────────────────────

class APIMetrics:
    """Armazena métricas operacionais da API em memória (thread-safe).

    Mantém contadores, histograma de latência e histórico recente de predições
    para alimentar endpoints de monitoramento e o dashboard.
    """

    def __init__(self, history_size: int = 500):
        self._lock = Lock()
        self.start_time: float = time.time()

        # Contadores
        self.total_requests: int = 0
        self.total_predictions: int = 0
        self.total_errors: int = 0
        self.total_drift_alerts: int = 0

        # Latência (em ms)
        self._latencies: deque = deque(maxlen=history_size)

        # Histórico de predições recentes
        self.prediction_history: deque = deque(maxlen=history_size)

        # Histórico de drift alerts
        self.drift_history: deque = deque(maxlen=history_size)

        # Distribuição de predições
        self.prediction_counts = {0: 0, 1: 0}

    # ---- helpers ----

    def record_request(self) -> str:
        """Registra início de request e retorna um request_id."""
        with self._lock:
            self.total_requests += 1
        return str(uuid.uuid4())[:8]

    def record_prediction(
        self,
        request_id: str,
        input_data: dict,
        prediction: int,
        label: str,
        latency_ms: float,
        drift_alerts: dict,
    ) -> None:
        with self._lock:
            self.total_predictions += 1
            self._latencies.append(latency_ms)
            self.prediction_counts[prediction] = (
                self.prediction_counts.get(prediction, 0) + 1
            )

            entry = {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction": prediction,
                "label": label,
                "latency_ms": round(latency_ms, 2),
                "drift_detected": bool(drift_alerts),
                "input_summary": {
                    k: v for k, v in input_data.items()
                    if isinstance(v, (int, float))
                },
            }
            self.prediction_history.append(entry)

            if drift_alerts:
                self.total_drift_alerts += 1
                self.drift_history.append({
                    "request_id": request_id,
                    "timestamp": entry["timestamp"],
                    "features": list(drift_alerts.keys()),
                    "details": drift_alerts,
                })

    def record_error(self, request_id: str, error: str) -> None:
        with self._lock:
            self.total_errors += 1

    def get_summary(self) -> dict:
        """Retorna resumo consolidado das métricas."""
        with self._lock:
            uptime_seconds = time.time() - self.start_time

            latencies = list(self._latencies)
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            else:
                avg_latency = min_latency = max_latency = p95 = 0.0

            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "uptime_human": _seconds_to_human(uptime_seconds),
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "error_rate": (
                    round(self.total_errors / max(self.total_requests, 1) * 100, 2)
                ),
                "total_drift_alerts": self.total_drift_alerts,
                "prediction_distribution": dict(self.prediction_counts),
                "latency_ms": {
                    "avg": round(avg_latency, 2),
                    "min": round(min_latency, 2),
                    "max": round(max_latency, 2),
                    "p95": round(p95, 2),
                },
                "recent_predictions": len(self.prediction_history),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def get_recent_predictions(self, n: int = 20) -> list:
        with self._lock:
            return list(self.prediction_history)[-n:]

    def get_drift_history(self, n: int = 50) -> list:
        with self._lock:
            return list(self.drift_history)[-n:]


def _seconds_to_human(s: float) -> str:
    """Converte segundos em string legível (ex: '2h 15m 30s')."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


# ──────────────────────────────────────────────
# 3. Detecção de drift
# ──────────────────────────────────────────────

DRIFT_THRESHOLD = 0.3  # 30%


def check_drift(
    input_df: pd.DataFrame,
    train_stats: Optional[dict],
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """Compara médias da entrada com estatísticas de treino.

    Retorna dict vazio se não houver drift ou se train_stats for None.
    """
    if train_stats is None:
        return {}

    drifts = {}
    for col, stat in train_stats.items():
        if col in input_df.columns and isinstance(stat, (int, float)):
            current_val = float(input_df[col].mean())
            if stat != 0:
                diff = abs(current_val - stat)
                ratio = diff / abs(stat)
                if ratio > threshold:
                    drifts[col] = {
                        "train_mean": round(stat, 4),
                        "current_value": round(current_val, 4),
                        "relative_diff_pct": round(ratio * 100, 1),
                        "status": "DRIFT DETECTED",
                    }
    return drifts


def get_drift_log_summary(log_path: str, max_items: int = 50) -> dict:
    """Lê o arquivo de log e extrai sumário de alertas de drift."""
    if not os.path.exists(log_path):
        return {"alerts_count": 0, "latest_alerts": []}

    latest_alerts = []
    alerts_count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "drift_alert" in line.lower() or "DRIFT ALERT" in line:
                alerts_count += 1
                latest_alerts.append(line.strip())

    return {
        "alerts_count": alerts_count,
        "latest_alerts": latest_alerts[-max_items:],
    }


# ──────────────────────────────────────────────
# 4. Instância global de métricas
# ──────────────────────────────────────────────

metrics = APIMetrics()
