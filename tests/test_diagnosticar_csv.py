"""Testes para src/diagnosticar_csv_problematicos.py"""

import pytest
import csv
from pathlib import Path

from src.diagnosticar_csv_problematicos import diagnosticar_arquivo, ler_csv_flexivel


class TestDiagnosticarArquivo:
    def test_diagnostica_csv_com_virgula(self, tmp_path):
        f = tmp_path / "dados.csv"
        f.write_text("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n")
        # Não deve lançar exceção
        diagnosticar_arquivo(str(f))

    def test_diagnostica_csv_com_ponto_virgula(self, tmp_path):
        f = tmp_path / "dados.csv"
        f.write_text("col1;col2;col3\nA;B;C\n")
        diagnosticar_arquivo(str(f))


class TestLerCsvFlexivel:
    def test_le_csv_comma(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        df = ler_csv_flexivel(str(f))
        assert df is not None
        assert len(df) == 2
        assert len(df.columns) == 3

    def test_le_csv_semicolon(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a;b;c\n1;2;3\n4;5;6\n")
        df = ler_csv_flexivel(str(f))
        assert df is not None
        assert len(df) >= 1

    def test_le_csv_tab(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
        df = ler_csv_flexivel(str(f))
        assert df is not None

    def test_retorna_none_single_column(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("singlecolumn\nvalue1\nvalue2\n")
        df = ler_csv_flexivel(str(f))
        # Com uma coluna só, nenhuma estratégia retorna > 1 coluna
        # Vai para leitura manual que também pode retornar None ou 1 coluna
        assert df is None or len(df.columns) <= 1
