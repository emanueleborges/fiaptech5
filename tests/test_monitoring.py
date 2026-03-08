"""Testes para o módulo src/monitoring.py."""

import json
import logging
import os
import tempfile
import time

import pandas as pd
import pytest

from src.monitoring import (
    APIMetrics,
    JSONFormatter,
    check_drift,
    get_drift_log_summary,
    log_event,
    setup_logger,
    _seconds_to_human,
    DRIFT_THRESHOLD,
)


# ── JSONFormatter ──


class TestJSONFormatter:
    def test_format_basic(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["level"] == "INFO"
        assert result["message"] == "hello"
        assert "timestamp" in result

    def test_format_with_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="warn", args=(), exc_info=None,
        )
        record.extra_fields = {"event": "test_event", "key": "value"}
        result = json.loads(formatter.format(record))
        assert result["event"] == "test_event"
        assert result["key"] == "value"


# ── setup_logger ──


class TestSetupLogger:
    def test_creates_logger(self, tmp_path):
        log_dir = str(tmp_path / "test_logs")
        logger = setup_logger(name="test_setup", log_dir=log_dir)
        assert isinstance(logger, logging.Logger)
        assert os.path.isdir(log_dir)

    def test_no_duplicate_handlers(self, tmp_path):
        log_dir = str(tmp_path / "dup_logs")
        name = "test_dup_check"
        l1 = setup_logger(name=name, log_dir=log_dir)
        h_count = len(l1.handlers)
        l2 = setup_logger(name=name, log_dir=log_dir)
        assert len(l2.handlers) == h_count
        assert l1 is l2


# ── log_event ──


class TestLogEvent:
    def test_log_event_info(self, tmp_path):
        log_dir = str(tmp_path / "evt_logs")
        logger = setup_logger(name="test_evt", log_dir=log_dir)
        log_event(logger, "my_event", key1="val1")
        # Should not raise

    def test_log_event_error_level(self, tmp_path):
        log_dir = str(tmp_path / "evt2_logs")
        logger = setup_logger(name="test_evt2", log_dir=log_dir)
        log_event(logger, "err", level="error", detail="fail")
        # Should not raise


# ── APIMetrics ──


class TestAPIMetrics:
    def test_record_request(self):
        m = APIMetrics()
        rid = m.record_request()
        assert isinstance(rid, str)
        assert m.total_requests == 1

    def test_record_prediction(self):
        m = APIMetrics()
        rid = m.record_request()
        m.record_prediction(rid, {"INDE": 5}, 1, "em_risco", 12.5, {})
        assert m.total_predictions == 1
        assert m.prediction_counts[1] == 1

    def test_record_prediction_with_drift(self):
        m = APIMetrics()
        rid = m.record_request()
        drifts = {"INDE": {"status": "DRIFT DETECTED"}}
        m.record_prediction(rid, {"INDE": 5}, 0, "fora", 10.0, drifts)
        assert m.total_drift_alerts == 1
        assert len(m.drift_history) == 1

    def test_record_error(self):
        m = APIMetrics()
        rid = m.record_request()
        m.record_error(rid, "some error")
        assert m.total_errors == 1

    def test_get_summary(self):
        m = APIMetrics()
        rid = m.record_request()
        m.record_prediction(rid, {}, 0, "fora", 15.0, {})
        summary = m.get_summary()
        assert summary["total_requests"] == 1
        assert summary["total_predictions"] == 1
        assert summary["total_errors"] == 0
        assert "latency_ms" in summary
        assert summary["latency_ms"]["avg"] == 15.0
        assert "uptime_human" in summary

    def test_get_summary_empty(self):
        m = APIMetrics()
        summary = m.get_summary()
        assert summary["total_requests"] == 0
        assert summary["latency_ms"]["avg"] == 0.0

    def test_get_recent_predictions(self):
        m = APIMetrics()
        for i in range(5):
            rid = m.record_request()
            m.record_prediction(rid, {}, i % 2, "lbl", 10.0, {})
        preds = m.get_recent_predictions(3)
        assert len(preds) == 3

    def test_get_drift_history(self):
        m = APIMetrics()
        rid = m.record_request()
        m.record_prediction(rid, {}, 1, "risco", 10.0, {"INDE": {"status": "DRIFT"}})
        hist = m.get_drift_history(10)
        assert len(hist) == 1
        assert "INDE" in hist[0]["features"]


# ── check_drift ──


class TestCheckDrift:
    def test_no_drift(self):
        df = pd.DataFrame([{"INDE": 5.0, "IDA": 3.0}])
        stats = {"INDE": 5.0, "IDA": 3.0}
        result = check_drift(df, stats)
        assert result == {}

    def test_drift_detected(self):
        df = pd.DataFrame([{"INDE": 50.0}])
        stats = {"INDE": 5.0}  # 50 vs 5 → 900% variation
        result = check_drift(df, stats)
        assert "INDE" in result
        assert result["INDE"]["status"] == "DRIFT DETECTED"

    def test_none_train_stats(self):
        df = pd.DataFrame([{"INDE": 5.0}])
        result = check_drift(df, None)
        assert result == {}

    def test_custom_threshold(self):
        df = pd.DataFrame([{"INDE": 6.0}])
        stats = {"INDE": 5.0}  # 20% diff
        assert check_drift(df, stats, threshold=0.1) != {}
        assert check_drift(df, stats, threshold=0.5) == {}

    def test_skips_zero_stat(self):
        df = pd.DataFrame([{"INDE": 5.0}])
        stats = {"INDE": 0}
        assert check_drift(df, stats) == {}

    def test_skips_non_numeric_stat(self):
        df = pd.DataFrame([{"INDE": 5.0}])
        stats = {"INDE": "string_value"}
        assert check_drift(df, stats) == {}


# ── get_drift_log_summary ──


class TestGetDriftLogSummary:
    def test_no_file(self, tmp_path):
        result = get_drift_log_summary(str(tmp_path / "nope.log"))
        assert result["alerts_count"] == 0

    def test_with_drift_entries(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text(
            '{"event": "prediction"}\n'
            '{"event": "drift_alert", "details": "bad"}\n'
            '{"event": "prediction"}\n'
            '{"event": "drift_alert", "details": "worse"}\n'
        )
        result = get_drift_log_summary(str(log_file))
        assert result["alerts_count"] == 2
        assert len(result["latest_alerts"]) == 2

    def test_max_items_limit(self, tmp_path):
        log_file = tmp_path / "big.log"
        lines = ['{"event": "drift_alert"}\n'] * 100
        log_file.write_text("".join(lines))
        result = get_drift_log_summary(str(log_file), max_items=5)
        assert len(result["latest_alerts"]) == 5


# ── _seconds_to_human ──


class TestSecondsToHuman:
    def test_seconds_only(self):
        assert _seconds_to_human(45) == "45s"

    def test_minutes_seconds(self):
        assert _seconds_to_human(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert _seconds_to_human(3661) == "1h 1m 1s"

    def test_zero(self):
        assert _seconds_to_human(0) == "0s"


# ── DRIFT_THRESHOLD ──


def test_drift_threshold_value():
    assert DRIFT_THRESHOLD == 0.3
