import sys
import os

# Adiciona a raiz do projeto ao PYTHONPATH para resolver importacoes 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.preprocessing import load_data, clean_data, sample_data, prepare_real_data, load_csv_with_fallback, extract_data_from_pdf
from src.feature_engineering import create_features, select_features


def train_model(data_path: str, model_output_path: str, metrics_output_path: str = "models/metrics.json"):
    """Pipeline principal de treinamento. Salva o pipeline completo com transformacoes.

    - Cria um ColumnTransformer para colunas numéricas e categóricas
    - Salva o pipeline (transformer + estimator) em `model_output_path`
    - Salva métricas em `metrics_output_path`
    """
    # 1. Carregar e preparar dados reais
    if data_path.endswith('.pdf'):
        print("Detectado arquivo PDF. Extraindo dados...")
        df = extract_data_from_pdf(data_path)
    else:
        print("Tentando carregar como CSV...")
        df = load_csv_with_fallback(data_path)

    X, y = prepare_real_data(df)

    # 2. Engenharia de Features
    X = create_features(X)
    X = select_features(X)

    # 3. Separacao X e y
    target_col = 'risk_label'
    if target_col not in X.columns:
        raise ValueError(f"Coluna alvo '{target_col}' nao encontrada no dataframe.")

    X = X.drop(columns=[target_col])
    y = y[target_col]

    # Identificar colunas numericas e categoricas
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # 5. Definir transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    # 6. Pipeline completo
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])

    # 7. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 8. Treinamento
    clf.fit(X_train, y_train)

    # 9. Avaliacao simples
    from src.evaluate import evaluate_model
    metrics = evaluate_model(clf, X_test, y_test)

    # 10. Salvar pipeline
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Pipeline salvo em {model_output_path}")

    # 11. Salvar metricas
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 12. Salvar estatisticas de treino para monitoramento de drift
    stats = X_train.select_dtypes(include=['int64', 'float64']).mean().to_dict()
    stats_path = os.path.join(os.path.dirname(model_output_path), "train_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Estatisticas de treino salvas em {stats_path}")

    return metrics


if __name__ == "__main__":
    data_path = 'data/raw/DATASET_FIAP.csv'
    model_path = "models/model.pkl"
    metrics_path = "models/metrics.json"
    train_model(data_path, model_path, metrics_path)
