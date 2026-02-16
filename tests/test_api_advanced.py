import pytest
from fastapi.testclient import TestClient
from api.app import app
import os
import json
import joblib

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data

def test_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get('/metrics')
        assert response.status_code == 200
        data = response.json()
        assert 'accuracy' in data or 'error' in data

def test_drift_endpoint():
    with TestClient(app) as client:
        response = client.get('/drift')
        assert response.status_code == 200
        assert 'threshold' in response.json()

def test_predict_drift_detection():
    payload = {
        'INDE': 1.0, 
        'IDA': 1.0,
        'IEG': 1.0,
        'IAA': 1.0,
        'IPS': 1.0,
        'IPP': 1.0,
        'FASE': 0,
        'PEDRA': 'Quartzo'
    }
    with TestClient(app) as client:
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'drift_alerts' in data
        assert len(data['drift_alerts']) > 0

def test_predict_normal_data():
    payload = {
        'INDE': 7.5,
        'IDA': 8.0,
        'IEG': 7.0,
        'IAA': 8.5,
        'IPS': 7.0,
        'IPP': 7.5,
        'FASE': 2,
        'PEDRA': 'Ametista'
    }
    with TestClient(app) as client:
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        assert 'prediction' in response.json()
