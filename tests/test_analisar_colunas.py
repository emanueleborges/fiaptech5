"""Testes para src/analisar_colunas.py"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

# A função analisar_dados_extraidos lê de um arquivo fixo.
# Vamos testar com mock do pd.read_csv.
from src import analisar_colunas


class TestAnalisarDadosExtraidos:
    @patch("src.analisar_colunas.pd.read_csv")
    def test_analisa_dados(self, mock_read_csv, capsys):
        mock_df = pd.DataFrame({
            "INDE": [7.0, 8.0, np.nan],
            "NOTA_MAT": ["5.0", "abc", "7.0"],
            "nome": ["Ana", "Bob", "Carol"],
            "fonte": ["a.csv", "a.csv", "b.csv"],
            "ano_referencia": [2022, 2022, 2023],
        })
        mock_read_csv.return_value = mock_df

        df, relevant = analisar_colunas.analisar_dados_extraidos()
        assert df is not None
        assert "INDE" in relevant
        captured = capsys.readouterr()
        assert "ANÁLISE" in captured.out

    @patch("src.analisar_colunas.pd.read_csv")
    def test_sem_colunas_relevantes(self, mock_read_csv, capsys):
        mock_df = pd.DataFrame({
            "coluna_a": [1, 2],
            "coluna_b": [3, 4],
        })
        mock_read_csv.return_value = mock_df

        df, relevant = analisar_colunas.analisar_dados_extraidos()
        assert len(relevant) == 0
