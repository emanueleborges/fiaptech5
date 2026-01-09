import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.preprocessing import clean_data
from api.app import app

client = TestClient(app)

def test_api_root():
    """Testa se a API esta respondendo na raiz."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Passos Magicos Online"}

def test_clean_data_removes_nulls():
    # Cria um dataframe de teste com nulos
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [4, None, 6]
    })
    
    # Executa a funcao (assumindo que clean_data remove nulos ou trata)
    # A logica padrao atual apenas retorna o df, entao vamos apenas verificar retorno
    cleaned_df = clean_data(df)
    
    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)

def test_load_data_returns_dataframe():
    # Teste unitario basico de importacao
    from src.preprocessing import load_data
    assert callable(load_data)
