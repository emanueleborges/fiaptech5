# Datathon: Case Passos Mágicos - Previsão de Risco de Defasagem Escolar

Projeto de Machine Learning Engineering para estimar risco de defasagem escolar de estudantes da Associação Passos Mágicos, cobrindo ciclo completo de MLOps: treinamento, validação, serving via API, testes, monitoramento e empacotamento com Docker.

## 1) Visão Geral do Projeto

**Objetivo de negócio**

Identificar, de forma antecipada, estudantes com maior probabilidade de risco educacional para apoiar ações pedagógicas e psicopedagógicas mais assertivas.

**Solução proposta**

Pipeline de ML em Python com:

- pré-processamento e limpeza de dados;
- engenharia e seleção de atributos;
- treinamento e avaliação de modelo supervisionado;
- serialização com `joblib`;
- disponibilização via API FastAPI (`/predict`);
- monitoramento com logs e painel de drift.

**Métrica principal de avaliação**

`F1-score` (com apoio de `accuracy`, `precision`, `recall` e `roc_auc` quando disponível). O F1-score equilibra precisão e recall, sendo adequado quando o foco é reduzir falsos negativos de alunos em risco sem gerar excesso de falsos positivos.

**Stack tecnológica**

- Linguagem: Python 3.9+
- ML: scikit-learn, pandas, numpy
- API: FastAPI + Uvicorn
- Serialização: joblib
- Testes: pytest + pytest-cov
- Empacotamento: Docker
- Monitoramento: logging + dashboard simples de drift
- Deploy: local (Docker) e pronto para cloud (Cloud Run/AWS/Heroku)

## 2) Estrutura do Projeto (Diretórios e Arquivos)

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

## 3) Instruções de Deploy (como subir o ambiente)

### Pré-requisitos

- Python 3.9+
- pip
- Docker (opcional para execução containerizada)

### Executando com Docker (Recomendado)

1.  **Construir a imagem:**
    ```bash
    docker build -t passos-magicos-api .
    ```

2.  **Rodar o container:**
    ```bash
    docker run --rm -p 8000:8000 passos-magicos-api
    ```

### Executando localmente

1.  **Instalar dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Treinar o modelo (necessário antes de rodar a API):**
    ```bash
    python src/train.py
    ```

   Opcional (definindo caminhos):

    ```bash
    python src/train.py data/raw/DATASET_FIAP.csv models/model.pkl
    ```

3.  **Iniciar a API:**
    ```bash
    uvicorn api.app:app --reload
    ```

4.  **Rodar os testes e cobertura:**
    ```bash
    pytest --cov=src --cov=api tests/
    ```

## 4) Exemplos de Chamadas à API

#### Interface Gráfica (Swagger UI)
Para facilitar a execução e testes, a aplicação conta com uma interface nativa.
1. Suba a aplicação (`uvicorn api.app:app --reload` ou via Docker).
2. Acesse no navegador: **`http://localhost:8000/docs`**
3. Você verá uma interface interativa para testar o endpoint `/predict` sem precisar de código.

### Via terminal (cURL)
**Endpoint:** `POST /predict`

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

**Outros endpoints úteis**

- `GET /health` (status da aplicação e carregamento do modelo)
- `GET /metrics` (métricas do último treino)
- `GET /drift` (informações de monitoramento)
- `GET /drift/dashboard` (painel HTML simples de drift)

## 4.1) Passo a passo para testar a aplicacao e endpoints

### 1) Preparar o ambiente

```bash
pip install -r requirements.txt
```

### 2) Treinar o modelo (gera `model.pkl` e estatisticas)

```bash
python src/train.py data/raw/DATATHON.xlsx models/model.pkl
```

### 3) Subir a API localmente

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### 4) Testar endpoints via curl

**Health check**

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Metricas do ultimo treino**

```bash
curl -X GET "http://127.0.0.1:8000/metrics"
```

**Predicao**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"INDE\":7.5,\"IDA\":8.0,\"IEG\":7.0,\"IAA\":8.5,\"IPS\":7.0,\"IPP\":7.5,\"FASE\":2,\"PEDRA\":\"Ametista\"}"
```

**Monitoramento de drift**

```bash
curl -X GET "http://127.0.0.1:8000/drift"
```

**Painel de drift (HTML)**

Abra no navegador: `http://127.0.0.1:8000/drift/dashboard`

### 5) Rodar testes automatizados

```bash
pytest --cov=src --cov=api tests/
```

## 5) Etapas do Pipeline de Machine Learning

1. **Pré-processamento (`src/preprocessing.py`)**
    - carga de CSV/Excel com fallback de encoding/separador;
    - limpeza de duplicidades e padronização básica;
    - preparação de features/target com criação de `risk_label`.

2. **Engenharia de features (`src/feature_engineering.py`)**
    - criação de variáveis derivadas (ex.: média de desempenho, flag de baixo engajamento);
    - seleção de atributos para modelagem.

3. **Treinamento e validação (`src/train.py`)**
    - pipeline com `ColumnTransformer` + `RandomForestClassifier`;
    - split treino/teste e avaliação;
    - persistência do modelo e métricas.

4. **Avaliação (`src/evaluate.py`)**
    - cálculo de `accuracy`, `precision`, `recall`, `f1` e `roc_auc` quando possível.

5. **Pós-processamento e serving (`api/app.py`)**
    - inferência via API REST;
    - monitoramento de drift por comparação estatística e logs.

## 6) Monitoramento Contínuo

- Logs rotativos em `logs/api_monitor.log`.
- Detecção de drift no endpoint `/predict` comparando média do input com estatísticas de treino (`models/train_stats.json`).
- Painel simples em `/drift/dashboard` com total e últimos alertas.

## 7) Artefatos da Entrega

- Código-fonte modularizado neste repositório.
- Documentação neste `README.md`.
- API local: `http://localhost:8000` (subir com Docker ou localmente).
- Vídeo gerencial (até 5 minutos): inserir link final na entrega do grupo.
