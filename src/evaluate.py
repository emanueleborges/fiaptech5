import json
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(pipeline, X_test, y_test) -> Dict:
    """Calcula métricas básicas de classificação e retorna um dicionário.

    Se o problema for binário, calcula também AUC quando disponível.
    """
    y_pred = pipeline.predict(X_test)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Tentar AUC se o estimator tiver predict_proba
    try:
        if hasattr(pipeline, 'predict_proba') or hasattr(pipeline.named_steps.get('classifier'), 'predict_proba'):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    return metrics
