from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os
import logging
from datetime import datetime

# Configuracao de Logs
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "api_monitor.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Passos Magicos - Predicao de Risco Escolar", version="1.0.0")

# Carregar modelo ao iniciar
MODEL_PATH = "models/model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Modelo carregado com sucesso de {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            print(f"Erro ao carregar o modelo: {e}")
    else:
        logger.warning(f"Modelo nao encontrado em {MODEL_PATH}")
        print("Modelo nao encontrado no caminho especificado.")

class StudentData(BaseModel):
    # Definir os campos de entrada conforme o modelo
    # Exemplo:
    # nota_matematica: float
    # frequencia: float
    pass

@app.get("/")
def home():
    logger.info("Acesso ao endpoint raiz")
    return {"message": "API Passos Magicos Online"}

@app.post("/predict")
def predict(data: StudentData):
    if not model:
        logger.error("Tentativa de predicao sem modelo carregado")
        raise HTTPException(status_code=500, detail="Modelo nao carregado")
    
    try:
        # Converter entrada para DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Log dos dados de entrada (Monitoramento de Drift)
        logger.info(f"Input Data: {data.dict()}")
        
        # Realizar predicao
        prediction = model.predict(input_data)
        result = int(prediction[0])
        
        # Log da predicao
        logger.info(f"Prediction: {result}")
        
        return {"prediction": result}
    except Exception as e:
        logger.error(f"Erro na predicao: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
