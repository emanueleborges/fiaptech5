import joblib
import pandas as pd
import json
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score

def evaluate_model(model_path: str, test_data_path: str, metrics_output_path: str = "metrics.json"):
    """Avalia o modelo treinado com metricas padrao e salva em arquivo."""
    
    # Carregar modelo
    if not os.path.exists(model_path):
        print(f"Modelo nao encontrado em {model_path}")
        return

    model = joblib.load(model_path)
    
    # Carregar dados de teste
    if not os.path.exists(test_data_path):
        print(f"Dados de teste nao encontrados em {test_data_path}")
        return

    df = pd.read_csv(test_data_path)
    
    # Separar X e y (AJUSTAR: Deve corresponder as colunas usadas no treino)
    target_col = 'risk_label' # Placeholder - Alterar para a coluna real (ex: 'PONTO_VIRADA', 'INDE')
    
    if target_col not in df.columns:
        print(f"Coluna alvo '{target_col}' nao encontrada no dataset de teste.")
        return

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    # Predicoes
    predictions = model.predict(X_test)
    
    # Calculo de Metricas
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "recall": recall,
        "precision": precision
    }

    # Exibir no Console
    print("=== Relatorio de Avaliacao ===")
    print(f"Acuracia: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nRelatorio Detalhado:\n", classification_report(y_test, predictions))
    
    # Salvar metricas
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metricas salvas em {metrics_output_path}")

if __name__ == "__main__":
    evaluate_model("models/model.pkl", "data/processed/test_data.csv")

