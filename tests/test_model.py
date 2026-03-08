"""Testes do modelo e da API."""

import pytest
import os
import json
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from src.preprocessing import sample_data
from src.feature_engineering import create_features, select_features
from src.train import train_model
from src.evaluate import evaluate_model


class TestModelTraining:
    """Testes de treinamento e avaliação do modelo."""

    def test_train_and_evaluate(self, tmp_path):
        data_path = tmp_path / "data.csv"
        data_path.write_text("risk_label,INDE,IDA,IPP,IEG\n1,8.0,7.0,7.5,9.0\n0,4.0,5.0,5.5,4.0")

        model_path = tmp_path / "model.pkl"
        metrics_path = tmp_path / "metrics.json"

        metrics = train_model(
            data_path=str(data_path),
            model_output_path=str(model_path),
            metrics_output_path=str(metrics_path),
        )

        assert os.path.exists(model_path)
        assert os.path.exists(metrics_path)
        assert "accuracy" in metrics

        pipeline = joblib.load(model_path)
        df = sample_data(20)
        df = create_features(df)
        df = select_features(df)

        X = df.drop(columns=["risk_label"])
        y = df["risk_label"]
        eval_metrics = evaluate_model(pipeline, X, y)
        assert "f1" in eval_metrics

    def test_sample_data(self):
        df = sample_data(10)
        assert len(df) == 10
        assert "risk_label" in df.columns


class TestModelAPI:
    """Testes dos endpoints da API relacionados ao modelo."""

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "model_loaded" in data

    def test_metrics_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "accuracy" in data or "error" in data

    def test_drift_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/drift")
            assert response.status_code == 200
            assert "threshold" in response.json()

    def test_predict_normal_data(self):
        payload = {
            "INDE": 7.5,
            "IDA": 8.0,
            "IEG": 7.0,
            "IAA": 8.5,
            "IPS": 7.0,
            "IPP": 7.5,
            "IPV": 0.8,
            "IAN": 7.5,
            "FASE": 2,
            "PEDRA": "Ametista",
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert "prediction" in response.json()

    def test_predict_drift_detection(self):
        payload = {
            "INDE": 1.0,
            "IDA": 1.0,
            "IEG": 1.0,
            "IAA": 1.0,
            "IPS": 1.0,
            "IPP": 1.0,
            "IPV": 0.5,
            "IAN": 5.0,
            "FASE": 0,
            "PEDRA": "Quartzo",
        }
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "drift_alerts" in data

    def test_train_endpoint(self):
        """Testa o endpoint /train que executa a pipeline completa de treinamento."""
        with TestClient(app) as client:
            response = client.post("/train", json={"n_samples": 50})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "metrics" in data
            assert "f1" in data["metrics"]
            assert data["model_reloaded"] is True
            assert "training_time_seconds" in data

    def test_train_endpoint_default(self):
        """Testa /train sem body (usa defaults)."""
        with TestClient(app) as client:
            response = client.post("/train")
            assert response.status_code == 200
            assert response.json()["status"] == "success"

    def test_monitoring_endpoint(self):
        """Testa o endpoint /monitoring com métricas operacionais."""
        with TestClient(app) as client:
            response = client.get("/monitoring")
            assert response.status_code == 200
            data = response.json()
            assert "total_requests" in data
            assert "uptime_seconds" in data
            assert "latency_ms" in data

    def test_monitoring_predictions_endpoint(self):
        """Testa o endpoint /monitoring/predictions."""
        with TestClient(app) as client:
            # Fazer uma predição primeiro
            client.post("/predict", json={
                "INDE": 7.5, "IDA": 8.0, "IEG": 7.0, "IAA": 8.5,
                "IPS": 7.0, "IPP": 7.5, "IPV": 0.8, "IAN": 7.5,
                "FASE": 2, "PEDRA": "Ametista",
            })
            response = client.get("/monitoring/predictions")
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "count" in data

    def test_drift_history_endpoint(self):
        """Testa o endpoint /drift/history."""
        with TestClient(app) as client:
            response = client.get("/drift/history")
            assert response.status_code == 200
            data = response.json()
            assert "total_alerts" in data
            assert "alerts" in data

    def test_drift_dashboard_html(self):
        """Testa o endpoint /drift/dashboard retorna HTML."""
        with TestClient(app) as client:
            response = client.get("/drift/dashboard")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "Dashboard" in response.text

    def test_predict_returns_request_id(self):
        """Testa que /predict retorna request_id e latency_ms."""
        with TestClient(app) as client:
            response = client.post("/predict", json={
                "INDE": 7.0, "IDA": 7.0, "IEG": 7.0, "IAA": 7.0,
                "IPS": 7.0, "IPP": 7.0, "IPV": 0.5, "IAN": 6.5,
                "FASE": 1, "PEDRA": "Topazio",
            })
            assert response.status_code == 200
            data = response.json()
            assert "request_id" in data
            assert "latency_ms" in data
            assert "label" in data
