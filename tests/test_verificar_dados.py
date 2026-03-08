"""Testes para src/analisar_colunas.py e src/verificar_dados_brutos.py"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from io import StringIO

from src.verificar_dados_brutos import verificar_arquivo


class TestVerificarArquivo:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("col1,col2\n1,2\n3,4\n5,6\n7,8\n9,10\n")
        # Não deve lançar exceção
        verificar_arquivo(str(f))

    def test_missing_file(self, tmp_path, capsys):
        verificar_arquivo(str(tmp_path / "nonexistent.csv"))
        captured = capsys.readouterr()
        assert "não encontrado" in captured.out

    def test_file_with_few_lines(self, tmp_path):
        f = tmp_path / "small.csv"
        f.write_text("a,b\n1,2\n")
        verificar_arquivo(str(f))


class TestVerificarMain:
    def test_main_with_existing_files(self, tmp_path, monkeypatch):
        from src.verificar_dados_brutos import main
        monkeypatch.chdir(tmp_path)
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True)
        proc = tmp_path / "data" / "processed"
        proc.mkdir(parents=True)
        for name in ["FILE_EXTRA_1.csv", "FILE_EXTRA_3.csv", "FILE_EXTRA_4.csv"]:
            (raw / name).write_text("a,b\n1,2\n")
        (proc / "dados_limpos.csv").write_text("x,y\n3,4\n")
        main()
