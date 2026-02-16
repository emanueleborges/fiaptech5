import pytest
from unittest.mock import patch, MagicMock
from src.download_data import download_data
import os

@patch('gdown.download')
@patch('os.makedirs')
@patch('os.path.exists')
def test_download_data(mock_exists, mock_makedirs, mock_gdown):
    # Simular que arquivos nao existem
    mock_exists.return_value = False
    
    # Chama a funcao
    download_data()
    
    # Verifica se gdown foi chamado (pelo menos uma vez)
    assert mock_gdown.called
