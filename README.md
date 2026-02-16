# Datathon: Case Passos Mágicos - Previsão de Risco de Defasagem Escolar

Este projeto faz parte do Datathon da Pós-Tech em Machine Learning Engineering da FIAP. O objetivo é desenvolver um modelo preditivo capaz de estimar o risco de defasagem escolar de estudantes da Associação Passos Mágicos, utilizando técnicas de MLOps para garantir um ciclo de vida completo e monitorável.

## 1. Visão Geral do Projeto
**Objetivo:** Identificar precocemente estudantes em vulnerabilidade educacional (risco de defasagem). O modelo utiliza dados socioeconômicos e de desempenho para classificar o risco, permitindo intervenções pedagógicas mais assertivas.

**Solução Proposta:** Uma pipeline automatizada que abrange pré-processamento, engenharia de features, treinamento via scikit-learn, API FastAPI e containerização Docker.

**Stack Tecnológica:**
*   **Linguagem:** Python 3.9
*   **ML Frameworks:** scikit-learn, pandas, numpy
*   **API:** FastAPI + Uvicorn
*   **Testes:** pytest + pytest-cov
*   **Containerização:** Docker
*   **Monitoramento:** Logging nativo + Endpoint de métricas

### 2) Estrutura do Projeto (Diretórios e Arquivos)

```bash
fiaptech5/
├── api/
│   ├── app.py                # Aplicação FastAPI e endpoints
├── data/
│   ├── raw/                  # Dados brutos (originais)
│   ├── processed/            # Dados limpos e preparados para treino
├── models/
│   ├── model.pkl             # Modelo treinado serializado
├── notebooks/                # Notebooks Jupyter para exploração e análise
├── src/
│   ├── preprocessing.py      # Funções de limpeza e tratamento inicial
│   ├── feature_engineering.py # Criação e seleção de variáveis
│   ├── train.py              # Pipeline de treinamento do modelo
│   ├── evaluate.py           # Avaliação de métricas
├── tests/
│   ├── test_pipeline.py      # Testes unitários
├── Dockerfile                # Arquivo para construção da imagem Docker
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação do projeto
```

### 3) Instruções de Deploy (como subir o ambiente)

#### Pré-requisitos
*   Docker instalado
*   Python 3.9+ (para execução local sem Docker)

#### Executando com Docker (Recomendado)

1.  **Construir a imagem:**
    ```bash
    docker build -t passos-magicos-api .
    ```

2.  **Rodar o container:**
    ```bash
    docker run -p 8000:8000 passos-magicos-api
    ```

#### Executando Localmente

1.  **Instalar dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Treinar o modelo (necessário antes de rodar a API):**
    ```bash
    python src/train.py
    ```

3.  **Iniciar a API:**
    ```bash
    uvicorn api.app:app --reload
    ```

4.  **Rodar os testes:**
    ```bash
    pytest
    ```

### 4) Exemplos de Chamadas à API

#### Interface Gráfica (Swagger UI)
Para facilitar a execução e testes, a aplicação conta com uma interface nativa.
1. Suba a aplicação (`uvicorn api.app:app --reload` ou via Docker).
2. Acesse no navegador: **`http://localhost:8000/docs`**
3. Você verá uma interface interativa para testar o endpoint `/predict` sem precisar de código.

#### Endpoints Disponíveis

**1. Health Check:** `GET /health`
```bash
curl http://localhost:8000/health
```
Resposta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-16T19:00:00.000000"
}
```

**2. Métricas do Modelo:** `GET /metrics`
```bash
curl http://localhost:8000/metrics
```
Resposta:
```json
{
  "accuracy": 0.995,
  "precision": 0.989,
  "recall": 1.0,
  "f1": 0.994,
  "roc_auc": 1.0
}
```

**3. Monitoramento de Drift:** `GET /drift`
```bash
curl http://localhost:8000/drift
```

**4. Predição de Risco:** `POST /predict`

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "INDE": 7.5,
  "IDA": 8.0,
  "IEG": 7.0,
  "IAA": 8.5,
  "IPS": 7.0,
  "IPP": 7.5,
  "FASE": 2,
  "PEDRA": "Ametista"
}'
```

**Exemplo de Resposta:**
```json
{
  "prediction": 0,
  "drift_alerts": {}
}
```

**Parâmetros esperados:**
- `INDE`: Índice do Desenvolvimento Educacional (0-10)
- `IDA`: Indicador de Aprendizagem (0-10)
- `IEG`: Indicador de Engajamento (0-10)
- `IAA`: Indicador de Autoavaliação (0-10)
- `IPS`: Indicador Psicossocial (0-10)
- `IPP`: Indicador Psicopedagógico (0-10)
- `FASE`: Fase do aluno (0-7)
- `PEDRA`: Classificação (Quartzo, Ametista, Topázio, Brilhante)

### 5) Etapas do Pipeline de Machine Learning

1.  **Pré-processamento dos Dados (`src/preprocessing.py`):** Carregamento de dados, remoção de linhas duplicadas, tratamento de valores nulos e tipagem de dados.
2.  **Engenharia de Features (`src/feature_engineering.py`):** Criação de novas variáveis baseadas em histórico acadêmico e indicadores socioeconômicos.
3.  **Treinamento (`src/train.py`):** Divisão de treino/teste, treinamento de um classificador (Random Forest) e serialização do modelo em `.pkl`.
4.  **Avaliação (`src/evaluate.py`):** Geração de métricas como Acurácia, Precision, Recall e F1-Score.

### 6) Testes Unitários e Cobertura

O projeto implementa testes unitários abrangentes para garantir a qualidade do código.

**Executar os testes:**
```bash
pytest
```

**Verificar cobertura de código:**
```bash
pytest --cov=src --cov=api --cov-report=term-missing
```

**Cobertura atual:** 84% (supera o requisito mínimo de 80%)

Os testes cobrem:
- Endpoints da API (health, metrics, drift, predict)
- Pipeline de pré-processamento
- Engenharia de features
- Treinamento e avaliação do modelo
- Detecção de drift

### 7) Monitoramento Contínuo

O sistema implementa as seguintes funcionalidades de monitoramento:

**Logs Avançados:**
- Sistema de logging com rotação automática de arquivos
- Localização: `logs/api_monitor.log`
- Registra todas as predições, erros e alertas de drift
- Formato: timestamp, nível, mensagem

**Detecção de Drift:**
- Monitoramento automático de desvio nas distribuições de entrada
- Comparação das médias dos dados de entrada vs. dados de treinamento
- Alerta quando a variação excede 30%
- Disponível via endpoint `/drift` e nos logs

**Métricas de Performance:**
- Endpoint `/metrics` retorna métricas de avaliação do modelo
- Métricas incluem: accuracy, precision, recall, F1-score, ROC-AUC
- Armazenadas em `models/metrics.json`

### 8) Dados e Modelo

**Dados de Treinamento:**
- O projeto utiliza dados sintéticos gerados baseados nos indicadores da Passos Mágicos
- Dados gerados em `data/processed/synthetic_data.csv`
- 1000 amostras com distribuições realistas

**Modelo Treinado:**
- Algoritmo: Random Forest Classifier
- Pipeline completo com pré-processamento integrado
- Localização: `models/model.pkl`
- Métricas de performance: >99% de acurácia no conjunto de teste
