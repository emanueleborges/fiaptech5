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
    model_output_path: str = "app/model/model.pkl",
    metrics_output_path: str = "models/metrics.json",
    train_stats_output_path: str = "models/train_stats.json",
    n_samples: int = 1000,
):
    """Treina o modelo conforme pipeline descrita no README.

    Etapas:
    1. Carrega dados de `data_path` (CSV/PDF) ou gera dataset sintético via sample_data
    2. Pré-processa e prepara os dados (preprocessing.py)
    3. Aplica engenharia de features (feature_engineering.py)
    4. Faz split treino/teste (80/20)
    5. Treina RandomForestClassifier com ColumnTransformer
    6. Avalia o modelo (evaluate.py) — F1-score como métrica principal
    7. Salva modelo (.pkl), métricas (.json) e estatísticas de treino (.json)
    """

    print("=" * 60)
    print("🚀 TREINAMENTO - PASSOS MÁGICOS")
    print("=" * 60)

    # ──────────────────────────────────────────
    # Etapa 1: Carregamento dos dados
    # ──────────────────────────────────────────
    df = None

    if data_path:
        print(f"\n📥 [Etapa 1] Carregando dados de {data_path}...")
        try:
            df = load_data(data_path)
            print(f"   ✅ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar {data_path}: {e}")
            print("   ↳ Usando dataset sintético como fallback...")
            df = None

    if df is None:
        print(f"\n📥 [Etapa 1] Gerando dataset sintético com {n_samples} amostras...")
        df = sample_data(n_samples)
        print(f"   ✅ Dataset sintético gerado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # ──────────────────────────────────────────
    # Etapa 2: Pré-processamento
    # ──────────────────────────────────────────
    print("\n🔧 [Etapa 2] Pré-processamento dos dados...")

    if "risk_label" in df.columns:
        print("   🎯 Coluna 'risk_label' encontrada — usando diretamente")
        df = create_features(df)
        df = select_features(df)
        X = df.drop(columns=["risk_label"])
        y = df["risk_label"].values
    else:
        print("   🎯 Preparando dados reais a partir de colunas de indicadores")
        X_raw, y_series = prepare_real_data(df)
        X = create_features(X_raw)
        X = select_features(X, target=None)
        y = y_series.values

    print(f"\n📊 Dados para modelagem:")
    print(f"   Features ({X.shape[1]}): {list(X.columns)}")
    print(f"   Amostras: {X.shape[0]}")
    print(f"   Target: {pd.Series(y).value_counts().to_dict()}")

    # ──────────────────────────────────────────
    # Etapa 3: Engenharia de features (já aplicada acima)
    # ──────────────────────────────────────────
    print("\n⚙️  [Etapa 3] Engenharia de features aplicada (avg_performance, low_engagement)")

    # ──────────────────────────────────────────
    # Etapa 4: Split treino/teste
    # ──────────────────────────────────────────
    print("\n✂️  [Etapa 4] Split treino/teste (80/20)...")

    # 2) Split treino/teste
    # Para conjuntos muito pequenos (ex.: apenas 1 amostra por classe),
    # a estratificação do scikit-learn falha. Nesses casos, fazemos split simples.
    value_counts = pd.Series(y).value_counts()
    can_stratify = len(value_counts) > 1 and value_counts.min() >= 2

    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print(f"\n   Treino: {len(X_train)} amostras")
    print(f"   Teste:  {len(X_test)} amostras")

    # ──────────────────────────────────────────
    # Etapa 5: Construção do pipeline (ColumnTransformer + RandomForest)
    # ──────────────────────────────────────────
    print("\n🏗️  [Etapa 5] Construindo pipeline (ColumnTransformer + RandomForestClassifier)...")

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

    # ──────────────────────────────────────────
    # Etapa 6: Treinamento
    # ──────────────────────────────────────────
    print("\n🎯 [Etapa 6] Treinando modelo...")
    pipeline.fit(X_train, y_train)

    # ──────────────────────────────────────────
    # Etapa 7: Avaliação (F1-score como métrica principal)
    # ──────────────────────────────────────────
    print("\n📊 [Etapa 7] Avaliando modelo (métrica principal: F1-score)...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print("\n   Métricas de avaliação:")
    for k, v in metrics.items():
        marker = " ⭐" if k == "f1" else ""
        print(f"   {k:>12s}: {v:.4f}{marker}")

    # Justificativa: usamos F1-score como métrica principal porque equilibra
    # precisão e recall — importante para não deixar alunos em risco sem atendimento
    # (falsos negativos) e ao mesmo tempo não sobrecarregar recursos com falsos positivos.
    print(f"\n   ✅ Modelo confiável — F1-score: {metrics.get('f1', 0):.4f}")
    print("   ↳ F1-score equilibra precisão e recall, ideal para detecção de risco educacional")

    # ──────────────────────────────────────────
    # Etapa 8: Salvamento dos artefatos (modelo + métricas + train_stats)
    # ──────────────────────────────────────────
    print("\n💾 [Etapa 8] Salvando artefatos de MLOps...")

    # ──────────────────────────────────────────
    # Etapa 8: Salvamento dos artefatos (modelo + métricas + train_stats)
    # ──────────────────────────────────────────
    print("\n💾 [Etapa 8] Salvando artefatos de MLOps...")

    model_path = PROJECT_ROOT / model_output_path
    metrics_path = PROJECT_ROOT / metrics_output_path
    train_stats_path = PROJECT_ROOT / train_stats_output_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    train_stats_path.parent.mkdir(parents=True, exist_ok=True)

    # 8a) Salvar pipeline completa (pré-processamento + modelo) com joblib
    joblib.dump(pipeline, model_path)

    # 8b) Salvar métricas de avaliação
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 8c) Estatísticas de treino para monitoramento de drift
    train_stats = {}
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    for col in X_train_df.select_dtypes(include=["int64", "float64"]).columns:
        train_stats[col] = float(X_train_df[col].mean())

    with open(train_stats_path, "w") as f:
        json.dump(train_stats, f, indent=2)

    print(f"   💾 Modelo salvo em:       {model_path}")
    print(f"   💾 Métricas salvas em:    {metrics_path}")
    print(f"   💾 Train stats salvas em: {train_stats_path}")
    print(f"\n{'=' * 60}")
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
    print(f"{'=' * 60}")

    return metrics


if __name__ == "__main__":
    # Suporta chamadas como:
    # python src/train.py
    # python src/train.py data/raw/DATASET_FIAP.csv app/model/model.pkl
    args = sys.argv[1:]

    data_arg = args[0] if len(args) >= 1 else None
    model_arg = args[1] if len(args) >= 2 else "app/model/model.pkl"

    train_model(data_path=data_arg, model_output_path=model_arg)