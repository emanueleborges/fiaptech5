"""Rotas e endpoints da API Passos Mágicos.

Endpoints (em ordem de execução lógica no Swagger):
1. POST /train              → Treinamento do modelo (pipeline completa)
2. GET  /health             → Status da API e modelo
3. GET  /metrics            → Métricas de performance do modelo (treino)
4. POST /predict            → Predição de risco escolar
5. GET  /monitoring         → Métricas operacionais (requests, latência, erros, drift)
6. GET  /monitoring/predictions → Histórico recente de predições
7. GET  /drift              → Status do monitoramento de drift
8. GET  /drift/history      → Histórico de alertas de drift (JSON)
9. GET  /drift/dashboard    → Painel HTML de monitoramento
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
import pandas as pd
import os
import json
import time
from datetime import datetime, timezone
from typing import Optional

from src.monitoring import (
    setup_logger,
    log_event,
    metrics,
    check_drift,
    get_drift_log_summary,
    DRIFT_THRESHOLD,
)

logger = setup_logger()

router = APIRouter()

# Referências globais injetadas pelo main.py
model = None
train_stats = None

METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics.json")
TRAIN_STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "train_stats.json")
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "api_monitor.log")


class StudentData(BaseModel):
    """Modelo de dados do aluno para predição."""
    INDE: float = 0.0
    IDA: float = 0.0
    IEG: float = 0.0
    IAA: float = 0.0
    IPS: float = 0.0
    IPP: float = 0.0
    FASE: int = 0
    PEDRA: str = "Quartzo"
    IAN: float = 0.0
    IPV: float = 0.0

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "INDE": 0, "IDA": 0, "IEG": 0, "IAA": 0,
                "IPS": 0, "IPP": 0, "FASE": 0, "IAN": 0,
                "IPV": 0, "PEDRA": "Quartzo",
            }
        },
    )


def _to_dataframe(obj: dict) -> pd.DataFrame:
    """Converte dict de entrada em DataFrame single-row."""
    if not obj:
        return pd.DataFrame()
    return pd.DataFrame([obj])


# ──────────────────────────────────────────
# 1. Endpoint de Treinamento (Pipeline Completa)
# ──────────────────────────────────────────

class TrainRequest(BaseModel):
    """Parâmetros opcionais para treinamento do modelo."""
    data_path: Optional[str] = None
    n_samples: int = 1000

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_path": None,
                "n_samples": 1000,
            }
        },
    )


@router.post("/train")
def train_model_endpoint(params: TrainRequest = TrainRequest()):
    """Executa a pipeline completa de treinamento do modelo.

    Etapas executadas:
    1. Carregamento dos dados (CSV ou sintético)
    2. Pré-processamento e limpeza
    3. Engenharia de features
    4. Split treino/teste (80/20)
    5. Treinamento (RandomForestClassifier)
    6. Avaliação (F1-score como métrica principal)
    7. Salvamento de artefatos (model.pkl, metrics.json, train_stats.json)
    8. Recarregamento automático do modelo na API
    """
    global model, train_stats

    t0 = time.time()
    log_event(logger, "train_started",
              data_path=params.data_path,
              n_samples=params.n_samples)

    try:
        from src.train import train_model

        train_metrics = train_model(
            data_path=params.data_path,
            n_samples=params.n_samples,
        )

        # Recarregar modelo e train_stats após treino
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)

        if os.path.exists(TRAIN_STATS_PATH):
            with open(TRAIN_STATS_PATH, "r") as f:
                train_stats = json.load(f)

        latency_s = time.time() - t0

        log_event(logger, "train_completed",
                  metrics=train_metrics,
                  latency_s=round(latency_s, 2))

        return {
            "status": "success",
            "message": "Modelo treinado e recarregado com sucesso",
            "metrics": train_metrics,
            "training_time_seconds": round(latency_s, 2),
            "model_reloaded": model is not None,
            "train_stats_reloaded": train_stats is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        latency_s = time.time() - t0
        log_event(logger, "train_error", level="error",
                  error=str(e), latency_s=round(latency_s, 2))
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")


# ──────────────────────────────────────────
# 2. Endpoints de saúde e métricas
# ──────────────────────────────────────────

@router.get("/health")
def health():
    """Retorna status da API, modelo e estatísticas de monitoramento."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "train_stats_loaded": train_stats is not None,
        "drift_threshold": f"{DRIFT_THRESHOLD * 100:.0f}%",
        "total_predictions": metrics.total_predictions,
        "total_drift_alerts": metrics.total_drift_alerts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
def get_metrics():
    """Retorna métricas de performance do modelo salvas durante o treino."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"error": "Metricas nao encontradas"}


# ──────────────────────────────────────────
# 3. Endpoint de predição
# ──────────────────────────────────────────

@router.post("/predict")
def predict(data: StudentData):
    """Realiza predição de risco escolar com monitoramento integrado."""
    request_id = metrics.record_request()
    t0 = time.time()

    if not model:
        metrics.record_error(request_id, "model_not_loaded")
        log_event(logger, "prediction_error", level="error",
                  request_id=request_id, reason="model_not_loaded")
        raise HTTPException(status_code=500, detail="Modelo nao carregado")

    try:
        dict_data = data.model_dump()
        input_data = _to_dataframe(dict_data)

        # ── Monitoramento de Drift ──
        drifts = check_drift(input_data, train_stats)
        if drifts:
            log_event(logger, "drift_alert", level="warning",
                      request_id=request_id,
                      drifts=drifts,
                      input_summary={k: v for k, v in dict_data.items() if isinstance(v, (int, float))})

        # ── Engenharia de features ──
        from src.feature_engineering import create_features, select_features
        input_data = create_features(input_data)
        input_data = select_features(input_data, target='risk_label')

        # ── Predição ──
        prediction = model.predict(input_data)
        result = int(prediction[0])
        label = "em_grupo_de_risco" if result == 1 else "fora_do_grupo_de_risco"

        latency_ms = (time.time() - t0) * 1000

        # ── Registrar métricas ──
        metrics.record_prediction(
            request_id=request_id,
            input_data=dict_data,
            prediction=result,
            label=label,
            latency_ms=latency_ms,
            drift_alerts=drifts,
        )

        # ── Log estruturado ──
        log_event(logger, "prediction",
                  request_id=request_id,
                  prediction=result,
                  label=label,
                  latency_ms=round(latency_ms, 2),
                  drift_detected=bool(drifts))

        return {
            "request_id": request_id,
            "prediction": result,
            "label": label,
            "latency_ms": round(latency_ms, 2),
            "drift_alerts": drifts,
        }

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        metrics.record_error(request_id, str(e))
        log_event(logger, "prediction_exception", level="error",
                  request_id=request_id, error=str(e),
                  latency_ms=round(latency_ms, 2))
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────
# 4. Endpoints de monitoramento operacional
# ──────────────────────────────────────────

@router.get("/monitoring")
def get_monitoring():
    """Retorna métricas operacionais da API (requests, latência, erros, drift)."""
    return metrics.get_summary()


@router.get("/monitoring/predictions")
def get_recent_predictions(n: int = 20):
    """Retorna as últimas N predições realizadas."""
    return {
        "count": min(n, len(metrics.prediction_history)),
        "predictions": metrics.get_recent_predictions(n),
    }


# ──────────────────────────────────────────
# 5. Endpoints de drift
# ──────────────────────────────────────────

@router.get("/drift")
def get_drift():
    """Retorna configuração e status do monitoramento de drift."""
    summary = metrics.get_summary()
    return {
        "status": "active",
        "threshold": f"{DRIFT_THRESHOLD * 100:.0f}% de variacao na media",
        "total_drift_alerts": summary["total_drift_alerts"],
        "total_predictions_monitored": summary["total_predictions"],
        "drift_rate": (
            round(summary["total_drift_alerts"] / max(summary["total_predictions"], 1) * 100, 2)
        ),
    }


@router.get("/drift/history")
def drift_history(n: int = 50):
    """Retorna histórico de alertas de drift em JSON."""
    history = metrics.get_drift_history(n)
    return {
        "total_alerts": metrics.total_drift_alerts,
        "showing": len(history),
        "alerts": history,
    }


@router.get("/drift/dashboard", response_class=HTMLResponse)
def drift_dashboard():
    """Painel HTML de monitoramento com métricas operacionais e drift."""
    summary = metrics.get_summary()
    log_summary = get_drift_log_summary(LOG_PATH)
    drift_hist = metrics.get_drift_history(20)
    recent_preds = metrics.get_recent_predictions(10)

    # Montar linhas de alertas de drift
    drift_rows = ""
    for d in reversed(drift_hist):
        features = ", ".join(d.get("features", []))
        drift_rows += f"""
        <tr>
            <td>{d.get('timestamp', '—')[:19]}</td>
            <td>{d.get('request_id', '—')}</td>
            <td style="color: #e74c3c; font-weight: bold;">{features}</td>
        </tr>"""

    # Montar linhas de predições recentes
    pred_rows = ""
    for p in reversed(recent_preds):
        label_color = "#e74c3c" if p["prediction"] == 1 else "#27ae60"
        drift_badge = "⚠️" if p.get("drift_detected") else "✅"
        pred_rows += f"""
        <tr>
            <td>{p.get('timestamp', '—')[:19]}</td>
            <td>{p.get('request_id', '—')}</td>
            <td style="color: {label_color}; font-weight: bold;">{p.get('label', '—')}</td>
            <td>{p.get('latency_ms', '—')} ms</td>
            <td>{drift_badge}</td>
        </tr>"""

    # Cor do erro rate
    error_rate = summary.get("error_rate", 0)
    error_color = "#27ae60" if error_rate < 1 else "#e67e22" if error_rate < 5 else "#e74c3c"

    # Extrair valores para uso no template
    latency = summary.get("latency_ms", {})
    avg_latency = latency.get("avg", 0)
    p95_latency = latency.get("p95", 0)
    pred_dist = summary.get("prediction_distribution", {})
    pred_dist_0 = pred_dist.get(0, 0)
    pred_dist_1 = pred_dist.get(1, 0)

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="30">
        <title>Dashboard de Monitoramento - Passos Mágicos</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 24px; }}
            h1 {{ color: #2c3e50; margin-bottom: 8px; }}
            .subtitle {{ color: #7f8c8d; margin-bottom: 24px; }}
            .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
            .card {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
            .card .value {{ font-size: 2em; font-weight: 700; color: #3498db; }}
            .card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 4px; }}
            .card.alert .value {{ color: #e74c3c; }}
            .card.success .value {{ color: #27ae60; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 32px; }}
            th {{ background: #3498db; color: #fff; padding: 12px 16px; text-align: left; font-weight: 600; }}
            td {{ padding: 10px 16px; border-bottom: 1px solid #ecf0f1; }}
            tr:hover {{ background: #f8f9fa; }}
            .section-title {{ font-size: 1.2em; font-weight: 600; margin: 24px 0 12px; color: #2c3e50; }}
            .footer {{ color: #95a5a6; font-size: 0.8em; margin-top: 32px; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>📊 Dashboard de Monitoramento</h1>
        <p class="subtitle">Passos Mágicos — Predição de Risco Escolar | Auto-refresh: 30s</p>

        <div class="cards">
            <div class="card success">
                <div class="value">{summary.get('uptime_human', '—')}</div>
                <div class="label">Uptime</div>
            </div>
            <div class="card">
                <div class="value">{summary.get('total_requests', 0)}</div>
                <div class="label">Total Requests</div>
            </div>
            <div class="card">
                <div class="value">{summary.get('total_predictions', 0)}</div>
                <div class="label">Predições</div>
            </div>
            <div class="card {'alert' if summary.get('total_errors', 0) > 0 else 'success'}">
                <div class="value">{summary.get('total_errors', 0)}</div>
                <div class="label">Erros</div>
            </div>
            <div class="card {'alert' if summary.get('total_drift_alerts', 0) > 0 else ''}">
                <div class="value">{summary.get('total_drift_alerts', 0)}</div>
                <div class="label">Drift Alerts</div>
            </div>
            <div class="card">
                <div class="value" style="color: {error_color}">{error_rate}%</div>
                <div class="label">Taxa de Erro</div>
            </div>
            <div class="card">
                <div class="value">{avg_latency} ms</div>
                <div class="label">Latência Média</div>
            </div>
            <div class="card">
                <div class="value">{p95_latency} ms</div>
                <div class="label">Latência P95</div>
            </div>
        </div>

        <div class="section-title">🔮 Distribuição de Predições</div>
        <div class="cards" style="grid-template-columns: repeat(2, 1fr); max-width: 400px;">
            <div class="card success">
                <div class="value">{pred_dist_0}</div>
                <div class="label">Fora de Risco (0)</div>
            </div>
            <div class="card alert">
                <div class="value">{pred_dist_1}</div>
                <div class="label">Em Risco (1)</div>
            </div>
        </div>

        <div class="section-title">📋 Predições Recentes</div>
        <table>
            <tr><th>Timestamp</th><th>Request ID</th><th>Resultado</th><th>Latência</th><th>Drift</th></tr>
            {pred_rows if pred_rows else '<tr><td colspan="5" style="text-align:center; color:#95a5a6;">Nenhuma predição registrada ainda</td></tr>'}
        </table>

        <div class="section-title">⚠️ Alertas de Drift (Threshold: {DRIFT_THRESHOLD * 100:.0f}%)</div>
        <table>
            <tr><th>Timestamp</th><th>Request ID</th><th>Features Afetadas</th></tr>
            {drift_rows if drift_rows else '<tr><td colspan="3" style="text-align:center; color:#95a5a6;">Nenhum alerta de drift registrado</td></tr>'}
        </table>

        <div class="section-title">📄 Alertas nos Logs ({log_summary['alerts_count']} total)</div>
        <table>
            <tr><th>Entrada de Log</th></tr>
            {''.join(f'<tr><td style="font-family:monospace;font-size:0.85em;">{a[:200]}</td></tr>' for a in log_summary['latest_alerts'][-10:]) or '<tr><td style="text-align:center; color:#95a5a6;">Nenhum alerta nos logs</td></tr>'}
        </table>

        <div class="footer">
            Atualizado em: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
            | <a href="/health">/health</a>
            | <a href="/metrics">/metrics</a>
            | <a href="/monitoring">/monitoring</a>
            | <a href="/drift">/drift</a>
            | <a href="/docs">/docs</a>
        </div>
    </body>
    </html>
    """
    return html
