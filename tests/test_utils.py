"""Testes para src/utils.py - funções auxiliares."""

import json
import os
import pytest
import pandas as pd

from src.utils import (
    get_project_root,
    load_json,
    save_json,
    to_dataframe,
    setup_logging,
    log_event,
)


class TestGetProjectRoot:
    def test_returns_path(self):
        root = get_project_root()
        assert root.exists()
        assert (root / "src").exists()

    def test_returns_pathlib_path(self):
        from pathlib import Path
        root = get_project_root()
        assert isinstance(root, Path)


class TestLoadJson:
    def test_loads_valid_json(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"a": 1, "b": 2}')
        result = load_json(str(p))
        assert result == {"a": 1, "b": 2}

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/nonexistent_file_xyz.json")


class TestSaveJson:
    def test_saves_and_creates_dir(self, tmp_path):
        p = tmp_path / "subdir" / "out.json"
        save_json({"x": 42}, str(p))
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["x"] == 42

    def test_handles_unicode(self, tmp_path):
        p = tmp_path / "unicode.json"
        save_json({"nome": "João"}, str(p))
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["nome"] == "João"


class TestToDataframe:
    def test_empty_dict(self):
        df = to_dataframe({})
        assert df.empty

    def test_single_row(self):
        df = to_dataframe({"a": 1, "b": 2})
        assert len(df) == 1
        assert df.iloc[0]["a"] == 1

    def test_preserves_types(self):
        df = to_dataframe({"num": 3.14, "text": "hello"})
        assert df["num"].dtype == float
        assert df["text"].iloc[0] == "hello"


class TestSetupLogging:
    def test_returns_logger(self, tmp_path):
        logger = setup_logging(log_dir=str(tmp_path), log_file="test.log")
        assert logger is not None
        assert logger.name == "PassosMagicosAPI"

    def test_creates_log_dir(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        setup_logging(log_dir=str(log_dir))
        assert log_dir.exists()


class TestLogEvent:
    def test_log_event_runs(self, tmp_path):
        # Apenas verifica que não dá erro
        log_event("test_event", status="success", detail="ok")
