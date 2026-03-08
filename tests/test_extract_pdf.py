"""Testes para src/extract_pdf_data.py"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.extract_pdf_data import (
    extract_from_pdf_with_plumber,
    extract_from_zip,
    fix_csv_reading,
    analyze_extracted_data,
)


class TestFixCsvReading:
    def test_reads_comma_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        df = fix_csv_reading(f)
        assert df is not None
        assert len(df) == 2

    def test_reads_semicolon_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1;col2;col3\n1;2;3\n4;5;6\n")
        df = fix_csv_reading(f)
        assert df is not None
        assert len(df) >= 1

    def test_reads_tab_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n")
        df = fix_csv_reading(f)
        assert df is not None

    def test_fallback_manual_read(self, tmp_path):
        # Criar um arquivo que falha nos métodos padrão
        f = tmp_path / "weird.csv"
        lines = ["a|b|c\n"] + [f"{i}|{i+1}|{i+2}\n" for i in range(5)]
        f.write_text("".join(lines))
        df = fix_csv_reading(f)
        assert df is not None


class TestExtractFromPdfWithPlumber:
    @patch("src.extract_pdf_data.pdfplumber")
    def test_extracts_tables(self, mock_plumber, tmp_path):
        # Mock PDF
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [
            [["Nome", "Nota"], ["Ana", "8"], ["Bob", "7"]]
        ]
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_plumber.open.return_value = mock_pdf

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.7 fake content")

        result = extract_from_pdf_with_plumber(str(pdf_path), "2022")
        assert result is not None
        assert len(result) == 2
        assert "ano_referencia" in result.columns

    @patch("src.extract_pdf_data.pdfplumber")
    def test_returns_none_no_tables(self, mock_plumber, tmp_path):
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_plumber.open.return_value = mock_pdf

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.7")
        result = extract_from_pdf_with_plumber(str(pdf_path), "2023")
        assert result is None

    @patch("src.extract_pdf_data.pdfplumber")
    def test_handles_error(self, mock_plumber, tmp_path):
        mock_plumber.open.side_effect = Exception("corrupted")
        pdf_path = tmp_path / "bad.pdf"
        pdf_path.write_bytes(b"%PDF-1.7")
        result = extract_from_pdf_with_plumber(str(pdf_path), "2022")
        assert result is None


class TestExtractFromZip:
    def test_extracts_zip(self, tmp_path):
        import zipfile
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("doc.xml", "<root>test</root>")
        result = extract_from_zip(str(zip_path))
        assert result is None  # ZIP retorna None por design

    def test_handles_invalid_zip(self, tmp_path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a zip")
        result = extract_from_zip(str(bad_zip))
        assert result is None


class TestAnalyzeExtractedData:
    def test_analyzes_dataframe(self, capsys):
        df = pd.DataFrame({
            "INDE": [7.0, 8.0, np.nan],
            "NOTA_MAT": [5.0, 6.0, 7.0],
            "nome": ["Ana", "Bob", "Carol"],
            "fonte": ["a.csv", "a.csv", "b.csv"],
            "ano_referencia": [2022, 2022, 2023],
        })
        analyze_extracted_data(df)
        captured = capsys.readouterr()
        assert "Shape" in captured.out

    def test_handles_none(self):
        analyze_extracted_data(None)

    def test_handles_empty(self):
        analyze_extracted_data(pd.DataFrame())

    def test_with_null_columns(self, capsys):
        df = pd.DataFrame({
            "INDE": [7.0, np.nan, np.nan, np.nan, np.nan],
            "MEDIA": [5.0, 6.0, 7.0, 8.0, 9.0],
            "ALUNO": ["A", "B", "C", "D", "E"],
            "ANO": [2022, 2022, 2023, 2023, 2024],
            "TURMA": ["A", "B", "A", "B", "A"],
            "SERIE": ["1", "2", "3", "4", "5"],
        })
        analyze_extracted_data(df)
        captured = capsys.readouterr()
        assert "nulos" in captured.out


class TestProcessAllFiles:
    @patch("src.extract_pdf_data.pdfplumber")
    def test_process_all_files_no_data(self, mock_plumber, tmp_path, monkeypatch):
        """Testa process_all_files quando não há arquivos."""
        from src.extract_pdf_data import process_all_files
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "raw").mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        result = process_all_files()
        assert result is None

    @patch("src.extract_pdf_data.pdfplumber")
    def test_process_with_csv_files(self, mock_plumber, tmp_path, monkeypatch):
        """Testa process_all_files com arquivos CSV extras."""
        from src.extract_pdf_data import process_all_files
        monkeypatch.chdir(tmp_path)
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        # Criar um FILE_EXTRA
        extra = raw_dir / "FILE_EXTRA_1.csv"
        extra.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        result = process_all_files()
        assert result is not None
        assert len(result) > 0
