import os
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto esteja no sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import load_data, sample_data, prepare_real_data
from src.feature_engineering import create_features, select_features
from src.evaluate import evaluate_model


def train_model(
    data_path: str = None,
    model_output_path: str = "models/model.pkl",
    metrics_output_path: str = "models/metrics.json",
    train_stats_output_path: str = "models/train_stats.json",
):
    """Treina o modelo conforme pipeline descrita no README.

    - Carrega dados de `data_path` (CSV) ou gera dataset sintético com sample_data
    - Aplica engenharia de features (create_features/select_features)
    - Faz split treino/teste
    - Treina RandomForest com ColumnTransformer
    - Salva modelo, métricas e estatísticas de treino
    """

    print("=" * 60)
    print("🚀 TREINAMENTO - PASSOS MÁGICOS")
    print("=" * 60)

    # 1) Carregar dados brutos ou sintéticos
    if data_path:
        print(f"📥 Carregando dados de {data_path}...")
        df = load_data(data_path)
    else:
        print("📥 Nenhum caminho informado, usando dataset sintético...")
        df = sample_data(1000)

    # Se já houver coluna de alvo, usamos diretamente; caso contrário, preparamos
    if "risk_label" in df.columns:
        print("🎯 Usando coluna 'risk_label' já presente nos dados")
        df = create_features(df)
        df = select_features(df)
        X = df.drop(columns=["risk_label"])
        y = df["risk_label"].values
    else:
        print("🎯 Preparando dados reais a partir de colunas de indicadores")
        X_raw, y_series = prepare_real_data(df)
        X = create_features(X_raw)
        X = select_features(X, target=None)
        y = y_series.values

    print(f"\n📊 Dados para modelagem:")
    print(f"   Features: {X.shape}")
    print(f"   Target: {pd.Series(y).value_counts().to_dict()}")

    # 2) Split treino/teste
    # Para conjuntos muito pequenos (ex.: apenas 1 amostra por classe),
    # a estratificação do scikit-learn falha. Nesses casos, fazemos split simples.
    value_counts = pd.Series(y).value_counts()
    can_stratify = len(value_counts) > 1 and value_counts.min() >= 2

    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print(f"\n📊 Split:")
    print(f"   Treino: {len(X_train)}")
    print(f"   Teste: {len(X_test)}")

    # 3) Construir pipeline de pré-processamento + modelo
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # 4) Treinar
    print("\n🎯 Treinando modelo...")
    pipeline.fit(X_train, y_train)

    # 5) Avaliar
    print("\n📊 Avaliando modelo...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # 6) Salvar artefatos
    model_path = PROJECT_ROOT / model_output_path
    metrics_path = PROJECT_ROOT / metrics_output_path
    train_stats_path = PROJECT_ROOT / train_stats_output_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    train_stats_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Estatísticas de treino para monitoramento de drift
    train_stats = {}
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    for col in X_train_df.select_dtypes(include=["int64", "float64"]).columns:
        train_stats[col] = float(X_train_df[col].mean())

    with open(train_stats_path, "w") as f:
        json.dump(train_stats, f, indent=2)

    print(f"\n💾 Modelo salvo em: {model_path}")
    print(f"💾 Métricas salvas em: {metrics_path}")
    print(f"💾 Estatisticas de treino salvas em: {train_stats_path}")

    return metrics


if __name__ == "__main__":
    # Suporta chamadas como:
    # python src/train.py
    # python src/train.py data/raw/DATASET_FIAP.csv models/model.pkl
    args = sys.argv[1:]

    data_arg = args[0] if len(args) >= 1 else None
    model_arg = args[1] if len(args) >= 2 else "models/model.pkl"

    train_model(data_path=data_arg, model_output_path=model_arg)