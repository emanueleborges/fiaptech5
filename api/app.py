from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import uvicorn
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json

# Configuracao de Logs Avancada
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler(os.path.join(LOG_DIR, "api_monitor.log"), maxBytes=10**6, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("PassosMagicosAPI")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

app = FastAPI(title="Passos Magicos - Predicao de Risco Escolar", version="1.0.0")

# Carregar modelo e estatisticas de treino
MODEL_PATH = "models/model.pkl"
TRAIN_STATS_PATH = "models/train_stats.json"
model = None
train_stats = None

def _to_dataframe(obj: dict) -> pd.DataFrame:
    """Converte dict de entrada em DataFrame single-row, mantendo ordem por chaves."""
    if not obj:
        return pd.DataFrame()
    return pd.DataFrame([obj])

@app.on_event("startup")
def load_resources():
    global model, train_stats
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Modelo carregado com sucesso de {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            
    if os.path.exists(TRAIN_STATS_PATH):
        try:
            with open(TRAIN_STATS_PATH, "r") as f:
                train_stats = json.load(f)
            logger.info("Estatisticas de treino carregadas.")
        except Exception as e:
            logger.error(f"Erro ao carregar estatisticas: {e}")

class StudentData(BaseModel):
    # Campos base esperados para facilitar o uso, mas permite extras
    INDE: float = 0.0
    IDA: float = 0.0
    IEG: float = 0.0
    IAA: float = 0.0
    IPS: float = 0.0
    IPP: float = 0.0
    FASE: int = 0
    PEDRA: str = "Quartzo"

    model_config = ConfigDict(extra='allow')

def check_drift(input_df: pd.DataFrame):
    """Monitoramento basico de drift comparando as medias do input vs treino."""
    if train_stats is None:
        return {}
    
    drifts = {}
    for col, stat in train_stats.items():
        if col in input_df.columns and isinstance(stat, (int, float)):
            current_val = float(input_df[col].mean())
            if stat != 0:
                diff = abs(current_val - stat)
                ratio = diff / abs(stat)
                if ratio > 0.3: # Threshold de 30%
                    drifts[col] = {"train_mean": stat, "current_mean": current_val, "status": "DRIFT DETECTED"}
    return drifts

@app.get("/metrics")
def get_metrics():
    """Retorna metricas de performance salvas durante o treino."""
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"error": "Metricas nao encontradas"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "timestamp": datetime.now().isoformat()}

@app.get("/drift")
def get_drift():
    """Endpoint simplificado para checar drift nos logs (demonstrativo)."""
    return {"message": "Monitoramento de drift ativo. Verifique os logs para alertas em tempo real.", "threshold": "30% de variacao na media"}

@app.post("/predict")
def predict(data: StudentData):
    if not model:
        logger.error("Tentativa de predicao sem modelo carregado")
        raise HTTPException(status_code=500, detail="Modelo nao carregado")
    
    try:
        dict_data = data.model_dump()
        input_data = _to_dataframe(dict_data)
        
        # Monitoramento de Drift
        drifts = check_drift(input_data)
        if drifts:
            logger.warning(f"DRIFT ALERT: {drifts}")
        
        # Aplicar engenharia de features
        from src.feature_engineering import create_features, select_features
        input_data = create_features(input_data)
        input_data = select_features(input_data, target='risk_label')
        
        # Realizar predicao
        prediction = model.predict(input_data)
        result = int(prediction[0])
        
        # Log da predicao
        logger.info(f"Input: {dict_data} | Prediction: {result}")
        
        return {"prediction": result, "drift_alerts": drifts}
    except Exception as e:
        logger.error(f"Erro na predicao: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
