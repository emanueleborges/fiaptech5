from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
import joblib
import pandas as pd
import uvicorn
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# Carregar modelo e estatisticas de treino
MODEL_PATH = "models/model.pkl"
TRAIN_STATS_PATH = "models/train_stats.json"
LOG_PATH = os.path.join(LOG_DIR, "api_monitor.log")
model = None
train_stats = None

def _to_dataframe(obj: dict) -> pd.DataFrame:
    """Converte dict de entrada em DataFrame single-row, mantendo ordem por chaves."""
    if not obj:
        return pd.DataFrame()
    return pd.DataFrame([obj])


def _build_fallback_model():
    from src.preprocessing import sample_data
    from src.feature_engineering import create_features, select_features

    df = sample_data(n=200)
    df = create_features(df)
    df = select_features(df)
    X = df.drop(columns=["risk_label"])
    y = df["risk_label"].values

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def load_resources():
    global model, train_stats
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Modelo carregado com sucesso de {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            model = _build_fallback_model()
            logger.info("Modelo fallback carregado em memoria")
            
    if os.path.exists(TRAIN_STATS_PATH):
        try:
            with open(TRAIN_STATS_PATH, "r") as f:
                train_stats = json.load(f)
            logger.info("Estatisticas de treino carregadas.")
        except Exception as e:
            logger.error(f"Erro ao carregar estatisticas: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title="Passos Magicos - Predicao de Risco Escolar",
    version="1.0.0",
    lifespan=lifespan,
)

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


def get_drift_log_summary(max_items: int = 20):
    if not os.path.exists(LOG_PATH):
        return {"alerts_count": 0, "latest_alerts": []}

    latest_alerts = []
    alerts_count = 0
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "DRIFT ALERT:" in line:
                alerts_count += 1
                latest_alerts.append(line.strip())

    return {
        "alerts_count": alerts_count,
        "latest_alerts": latest_alerts[-max_items:],
    }

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


@app.get("/drift/dashboard", response_class=HTMLResponse)
def drift_dashboard():
        summary = get_drift_log_summary()
        html = f"""
        <html>
            <head><title>Painel de Drift - Passos Magicos</title></head>
            <body style=\"font-family: Arial, sans-serif; margin: 24px;\">
                <h2>Painel de Drift do Modelo</h2>
                <p><b>Threshold:</b> 30% de variacao na media</p>
                <p><b>Total de alertas:</b> {summary['alerts_count']}</p>
                <h3>Ultimos alertas</h3>
                <ul>
                    {''.join(f'<li>{a}</li>' for a in summary['latest_alerts'])}
                </ul>
            </body>
        </html>
        """
        return html

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
