import pytest
import pandas as pd
import os
import joblib
from src.preprocessing import clean_data, sample_data, load_data
from src.feature_engineering import create_features, select_features
from src.train import train_model
from src.evaluate import evaluate_model

def test_preprocessing_clean():
    df = pd.DataFrame({' A ': [1, 2, 2], 'B': [3, 4, 4]})
    cleaned = clean_data(df)
    assert 'a' in cleaned.columns
    assert len(cleaned) == 2

def test_sample_data():
    df = sample_data(10)
    assert len(df) == 10
    assert 'risk_label' in df.columns

def test_feature_engineering_creation():
    df = pd.DataFrame({
        'INDE': [8.0, 4.0],
        'IDA': [7.0, 5.0],
        'IPP': [7.5, 5.5],
        'IEG': [9.0, 4.0]
    })
    feat = create_features(df)
    assert 'avg_performance' in feat.columns
    assert 'low_engagement' in feat.columns
    assert feat.loc[1, 'low_engagement'] == 1

def test_train_and_evaluate(tmp_path):
    # Criar arquivo de dados sintético
    data_path = tmp_path / "data.csv"
    data_path.write_text("risk_label,INDE,IDA,IPP,IEG\n1,8.0,7.0,7.5,9.0\n0,4.0,5.0,5.5,4.0")

    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    # Treinar usando o arquivo de dados sintético
    metrics = train_model(data_path=str(data_path), model_output_path=str(model_path), metrics_output_path=str(metrics_path))

    assert os.path.exists(model_path)
    assert os.path.exists(metrics_path)
    assert 'accuracy' in metrics

    # Testar avaliação separadamente
    pipeline = joblib.load(model_path)
    df = sample_data(20)
    df = create_features(df)
    df = select_features(df)

    X = df.drop(columns=['risk_label'])
    y = df['risk_label']
    eval_metrics = evaluate_model(pipeline, X, y)
    assert 'f1' in eval_metrics

def test_load_data_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("a,b\n1,2")
    df = load_data(str(p))
    assert not df.empty
    assert df.iloc[0, 0] == 1
