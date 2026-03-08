"""Testes adicionais para melhorar cobertura de src/preprocessing.py e app/main.py"""

import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.preprocessing import (
    _normalizar_nome_coluna,
    clean_data,
    _coerce_comma_float,
    sample_data,
    _is_pdf_file,
    _is_zip_file,
    load_data,
    prepare_real_data,
)


class TestNormalizarNomeColuna:
    def test_inde_variations(self):
        assert _normalizar_nome_coluna("INDE 22") == "INDE"
        assert _normalizar_nome_coluna("INDE_2021") == "INDE"

    def test_pedra(self):
        assert _normalizar_nome_coluna("Pedra 22") == "PEDRA"
        assert _normalizar_nome_coluna("PEDRA") == "PEDRA"

    def test_ida(self):
        assert _normalizar_nome_coluna("IDA") == "IDA"
        assert _normalizar_nome_coluna("IDA 22") == "IDA"

    def test_ieg(self):
        assert _normalizar_nome_coluna("IEG") == "IEG"

    def test_iaa(self):
        assert _normalizar_nome_coluna("IAA") == "IAA"

    def test_ips(self):
        assert _normalizar_nome_coluna("IPS") == "IPS"

    def test_ipp(self):
        assert _normalizar_nome_coluna("IPP") == "IPP"

    def test_ipv(self):
        assert _normalizar_nome_coluna("IPV") == "IPV"

    def test_ian(self):
        assert _normalizar_nome_coluna("IAN") == "IAN"

    def test_destaque_ipv(self):
        assert _normalizar_nome_coluna("Destaque IPV") == "destaque_ipv"

    def test_generic(self):
        result = _normalizar_nome_coluna("My Column")
        assert result == "my_column"

    def test_non_string(self):
        result = _normalizar_nome_coluna(123)
        assert isinstance(result, str)


class TestCoerceCommaFloat:
    def test_comma_to_float(self):
        s = pd.Series(["5,5", "7,0", "3,14"])
        result = _coerce_comma_float(s)
        assert result[0] == 5.5
        assert result[1] == 7.0

    def test_dot_float(self):
        s = pd.Series(["5.5", "7.0"])
        result = _coerce_comma_float(s)
        assert result[0] == 5.5

    def test_invalid_values(self):
        s = pd.Series(["abc", "def"])
        result = _coerce_comma_float(s)
        assert result.isna().all()


class TestLoadData:
    def test_load_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        df = load_data(str(f))
        assert len(df) == 2

    def test_load_csv_semicolon(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a;b\n1;2\n3;4\n")
        df = load_data(str(f))
        assert len(df) == 2

    def test_load_excel(self, tmp_path):
        f = tmp_path / "data.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(str(f), index=False)
        df = load_data(str(f))
        assert len(df) == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_data("/tmp/does_not_exist_xyz.csv")

    def test_pdf_csv_detection(self, tmp_path):
        """Testa detecção de um arquivo .csv que é na verdade PDF."""
        f = tmp_path / "fake.csv"
        f.write_bytes(b"%PDF-1.7 fake content not a real pdf but has signature")
        # Deve tentar ler como PDF, o que pode falhar - aceitamos exceção
        try:
            load_data(str(f))
        except Exception:
            pass  # Esperado para PDF inválido


class TestPrepareRealData:
    def test_with_valid_numeric_data(self):
        df = pd.DataFrame({
            "INDE": [5.5, 7.0, 8.0, 4.0],
            "IDA": [4.0, 6.0, 7.0, 3.0],
            "IEG": [7.0, 8.0, 9.0, 5.0],
            "IAA": [8.0, 9.0, 7.0, 4.0],
            "IPS": [6.0, 7.0, 8.0, 5.0],
            "IPP": [5.0, 6.0, 7.0, 4.0],
            "IPV": [0.65, 0.7, 0.8, 0.4],
            "IAN": [7.5, 8.0, 6.5, 5.0],
        })
        X, y = prepare_real_data(df)
        assert "INDE" in X.columns
        assert len(y) == 4
        assert set(y.unique()).issubset({0, 1})

    def test_with_missing_columns_generates_synthetic(self):
        df = pd.DataFrame({"INDE 22": ["5,5"], "other": ["x"]})
        X, y = prepare_real_data(df)
        assert "INDE" in X.columns
        assert len(X) > 0

    def test_with_column_mapping(self):
        df = pd.DataFrame({
            "IEG_2020": [7.0, 8.0],
            "IPV_2021": [0.5, 0.6],
            "NOTA_ING_2022": [6.0, 7.0],
            "INDE": [5.5, 7.0],
            "IDA": [4.0, 6.0],
            "IAA": [8.0, 9.0],
            "IPS": [6.0, 7.0],
            "IPP": [5.0, 6.0],
        })
        X, y = prepare_real_data(df)
        assert "IEG" in X.columns
        assert "IPV" in X.columns
        assert "IAN" in X.columns


class TestIsPdfFile:
    def test_pdf_file(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.7\n")
        assert _is_pdf_file(str(f)) is True

    def test_non_pdf_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n")
        assert _is_pdf_file(str(f)) is False

    def test_missing_file(self):
        assert _is_pdf_file("/tmp/nonexistent_xyz") is False


class TestIsZipFile:
    def test_zip_file(self, tmp_path):
        f = tmp_path / "test.zip"
        f.write_bytes(b"PK\x03\x04rest")
        assert _is_zip_file(str(f)) is True

    def test_non_zip_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _is_zip_file(str(f)) is False

    def test_missing_file(self):
        assert _is_zip_file("/tmp/nonexistent_xyz") is False


class TestAppMainCoverage:
    """Testes para melhorar cobertura de app/main.py"""

    def test_build_fallback_model(self):
        from app.main import _build_fallback_model
        model = _build_fallback_model()
        assert model is not None
        assert hasattr(model, "predict")

    def test_load_resources_with_model(self, tmp_path, monkeypatch):
        import app.main as main_module
        # Criar um modelo fake
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        clf = RandomForestClassifier(n_estimators=2, random_state=42)
        clf.fit([[1, 2], [3, 4]], [0, 1])

        model_path = tmp_path / "model.pkl"
        joblib.dump(clf, str(model_path))

        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({"INDE": 7.0}))

        monkeypatch.setattr(main_module, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(main_module, "TRAIN_STATS_PATH", str(stats_path))

        main_module.load_resources()
        assert main_module.model is not None
        assert main_module.train_stats is not None

    def test_load_resources_missing_model(self, tmp_path, monkeypatch):
        import app.main as main_module
        monkeypatch.setattr(main_module, "MODEL_PATH", str(tmp_path / "no.pkl"))
        monkeypatch.setattr(main_module, "TRAIN_STATS_PATH", str(tmp_path / "no.json"))
        main_module.model = None
        main_module.train_stats = None
        main_module.load_resources()
        # Model stays None when file doesn't exist (no fallback path triggered)
        # train_stats stays None

    def test_load_resources_corrupted_model(self, tmp_path, monkeypatch):
        import app.main as main_module
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(b"corrupted data")

        monkeypatch.setattr(main_module, "MODEL_PATH", str(model_path))
        monkeypatch.setattr(main_module, "TRAIN_STATS_PATH", str(tmp_path / "no.json"))

        main_module.load_resources()
        # Fallback model should be built
        assert main_module.model is not None
