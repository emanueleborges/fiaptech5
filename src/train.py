import sys
import os

# Adiciona a raiz do projeto ao PYTHONPATH para resolver importacoes 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features, select_features
import os

def train_model(data_path: str, model_output_path: str):
    """Pipeline principal de treinamento."""
    
    # 1. Carregar dados
    df = load_data(data_path)
    
    # 2. Pre-processamento
    df = clean_data(df)
    
    # 3. Engenharia de Features
    df = create_features(df)
    df = select_features(df)
    
    # 4. Separacao X e y (Ajustar 'target' para o nome real da coluna alvo)
    target_col = 'risk_label' # Placeholder
    if target_col not in df.columns:
        print(f"Coluna alvo '{target_col}' nao encontrada.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 5. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Definicao do Modelo
    model = RandomForestClassifier(random_state=42)
    
    # 7. Treinamento
    model.fit(X_train, y_train)
    
    # 8. Salvar modelo
    joblib.dump(model, model_output_path)
    print(f"Modelo salvo em {model_output_path}")

if __name__ == "__main__":
    # Exemplo de execucao
    data_path = "data/processed/dados_consolidados.csv"
    model_path = "models/model.pkl"
    # train_model(data_path, model_path)
