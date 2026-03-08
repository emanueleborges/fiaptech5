"""Testes para src/monitor_drift.py"""

import json
import os
import pytest
import pandas as pd

from src.monitor_drift import (
    load_train_stats,
    compute_current_stats,
    compare_stats,
    generate_report,
)


class TestLoadTrainStats:
    def test_loads_valid_file(self, tmp_path):
        p = tmp_path / "stats.json"
        p.write_text(json.dumps({"INDE": 7.0, "IDA": 6.5}))
        result = load_train_stats(str(p))
        assert result["INDE"] == 7.0

    def test_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            load_train_stats("/tmp/missing_stats_xyz.json")


class TestComputeCurrentStats:
    def test_computes_means(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("INDE,IDA,nome\n7.0,6.0,Ana\n9.0,8.0,Bob\n")
        stats = compute_current_stats(str(csv_path))
        assert abs(stats["INDE"] - 8.0) < 0.01
        assert abs(stats["IDA"] - 7.0) < 0.01
        assert "nome" not in stats.index  # coluna não numérica

    def test_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            compute_current_stats("/tmp/missing_data_xyz.csv")


class TestCompareStats:
    def test_detects_drift(self):
        train = {"INDE": 7.0, "IDA": 6.0}
        current = pd.Series({"INDE": 2.0, "IDA": 6.1})
        drifts = compare_stats(train, current, threshold=0.3)
        assert "INDE" in drifts
        assert drifts["INDE"]["status"] == "DRIFT DETECTED"
        assert "IDA" not in drifts  # pouca diferença

    def test_no_drift(self):
        train = {"INDE": 7.0}
        current = pd.Series({"INDE": 7.1})
        drifts = compare_stats(train, current)
        assert len(drifts) == 0

    def test_skips_zero_train_mean(self):
        train = {"INDE": 0}
        current = pd.Series({"INDE": 5.0})
        drifts = compare_stats(train, current)
        assert len(drifts) == 0

    def test_skips_missing_column(self):
        train = {"INDE": 7.0, "UNKNOWN": 5.0}
        current = pd.Series({"INDE": 7.0})
        drifts = compare_stats(train, current)
        assert "UNKNOWN" not in drifts


class TestGenerateReport:
    def test_generates_report_file(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps({"INDE": 7.0, "IDA": 6.0}))

        csv_path = tmp_path / "input.csv"
        csv_path.write_text("INDE,IDA\n2.0,6.0\n3.0,5.5\n")

        output_dir = tmp_path / "reports"
        report_path = generate_report(
            str(csv_path), str(stats_path), str(output_dir)
        )

        assert os.path.exists(report_path)
        with open(report_path) as f:
            report = json.load(f)
        assert "drifts" in report
        assert "timestamp" in report
        assert "INDE" in report["drifts"]
