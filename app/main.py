"""Arquivo principal da API - Passos Mágicos.

Responsável por:
- Configuração do logger estruturado (JSON)
- Carregamento do modelo e estatísticas de treino
- Inicialização da aplicação FastAPI com monitoramento
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import uvicorn
import os
import json
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Garante que o diretório raiz do projeto esteja no PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import setup_logger, log_event, metrics  # noqa: E402

# ── Logger estruturado ──
logger = setup_logger()

# Caminhos dos modelos (agora em app/model/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
TRAIN_STATS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "train_stats.json")

model = None
train_stats = None


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

    # ── Carregar modelo ──
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            log_event(logger, "model_loaded", path=MODEL_PATH)
        except Exception as e:
            log_event(logger, "model_load_error", level="error", path=MODEL_PATH, error=str(e))
            model = _build_fallback_model()
            log_event(logger, "model_fallback_built", level="warning")
    else:
        log_event(logger, "model_not_found", level="warning", path=MODEL_PATH)
        model = _build_fallback_model()
        log_event(logger, "model_fallback_built", level="warning")

    # ── Carregar estatísticas de treino (para drift) ──
    if os.path.exists(TRAIN_STATS_PATH):
        try:
            with open(TRAIN_STATS_PATH, "r") as f:
                train_stats = json.load(f)
            log_event(logger, "train_stats_loaded", path=TRAIN_STATS_PATH)
        except Exception as e:
            log_event(logger, "train_stats_load_error", level="error",
                      path=TRAIN_STATS_PATH, error=str(e))
    else:
        log_event(logger, "train_stats_not_found", level="warning", path=TRAIN_STATS_PATH)

    # Injetar model e train_stats nas rotas
    import app.routes as routes_module
    routes_module.model = model
    routes_module.train_stats = train_stats

    log_event(logger, "api_startup_complete",
              model_loaded=model is not None,
              train_stats_loaded=train_stats is not None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_resources()
    yield
    log_event(logger, "api_shutdown")


app = FastAPI(
    title="Passos Magicos - Predicao de Risco Escolar",
    description=(
        "API para predição de risco de defasagem escolar com "
        "monitoramento de drift e logging estruturado."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Registrar rotas
from app.routes import router  # noqa: E402
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
