# Passos Mágicos - Datathon
## Predição de Risco de Defasagem Escolar

### 1) Visão Geral do Projeto
**Objetivo:** Este projeto visa desenvolver um modelo preditivo para estimar o risco de defasagem escolar de estudantes atendidos pela Associação Passos Mágicos. O objetivo é permitir que a instituição atue preventivamente, oferecendo suporte personalizado para evitar a evasão e melhorar o desempenho dos alunos.

**Solução Proposta:** Construção de uma pipeline completa de Machine Learning, modularizada e encapsulada, que vai desde o pré-processamento dos dados até o deploy de uma API para consumo do modelo em produção.

**Stack Tecnológica:**
*   **Linguagem:** Python 3.9+
*   **Frameworks de ML:** scikit-learn, pandas, numpy
*   **API:** FastAPI
*   **Serialização:** joblib
*   **Testes:** pytest
*   **Empacotamento:** Docker
*   **Monitoramento:** Logging básico configurado

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
