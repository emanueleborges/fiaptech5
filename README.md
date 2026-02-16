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

#### Via Terminal (cURL)
**Endpoint:** `POST /predict`

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "nota_matematica": 7.5,
  "frequencia": 0.95
}'
```

**Exemplo de Resposta:**
```json
{
  "prediction": 0
}
```

### 5) Etapas do Pipeline de Machine Learning

1.  **Pré-processamento dos Dados (`src/preprocessing.py`):** Carregamento de dados, remoção de linhas duplicadas, tratamento de valores nulos e tipagem de dados.
2.  **Engenharia de Features (`src/feature_engineering.py`):** Criação de novas variáveis baseadas em histórico acadêmico e indicadores socioeconômicos.
3.  **Treinamento (`src/train.py`):** Divisão de treino/teste, treinamento de um classificador (Random Forest) e serialização do modelo em `.pkl`.
4.  **Avaliação (`src/evaluate.py`):** Geração de métricas como Acurácia, Precision, Recall e F1-Score.
