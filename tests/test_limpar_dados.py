"""Testes para src/limpar_dados.py"""

import pytest
import pandas as pd
import numpy as np

from src.limpar_dados import limpar_nome_coluna, extrair_numero


class TestLimparNomeColuna:
    def test_remove_special_chars(self):
        assert limpar_nome_coluna("Col@Name!") == "ColName"

    def test_strips_whitespace(self):
        assert limpar_nome_coluna("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert limpar_nome_coluna("a   b   c") == "a b c"

    def test_non_string_input(self):
        result = limpar_nome_coluna(123)
        assert isinstance(result, str)

    def test_empty_string(self):
        assert limpar_nome_coluna("") == ""


class TestExtrairNumero:
    def test_float_value(self):
        assert extrair_numero(3.14) == 3.14

    def test_int_value(self):
        assert extrair_numero(42) == 42.0

    def test_string_with_number(self):
        assert extrair_numero("abc 7.5 xyz") == 7.5

    def test_string_with_comma(self):
        assert extrair_numero("5,5") == 5.5

    def test_nan_value(self):
        result = extrair_numero(np.nan)
        assert pd.isna(result)

    def test_no_number(self):
        result = extrair_numero("abc")
        assert pd.isna(result)

    def test_negative_number(self):
        assert extrair_numero("-3.2") == -3.2

    def test_none_like(self):
        result = extrair_numero(None)
        assert pd.isna(result)


class TestLimparDados:
    def test_limpar_dados_full(self, tmp_path, monkeypatch):
        """Testa a função limpar_dados() completa com mock de dados."""
        monkeypatch.chdir(tmp_path)
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)

        # Criar CSV com dados de teste incluindo keywords de notas
        df = pd.DataFrame({
            "INDE": ["5,5", "7,0", "8,0"],
            "IDA_NOTA": ["4,0", "6,0", "7,0"],
            "IEG": ["7,0", "8,0", "9,0"],
            "nome": ["Ana", "Bob", "Carol"],
        })
        df.to_csv(proc_dir / "dados_passos_completos.csv", index=False)

        from src.limpar_dados import limpar_dados
        result_df, colunas_notas = limpar_dados()
        assert result_df is not None
        assert len(colunas_notas) > 0
        # Deve ter criado dados_limpos.csv
        assert (proc_dir / "dados_limpos.csv").exists()
