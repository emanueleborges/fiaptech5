# Roteiro Mínimo: Treinar, Subir API e Testar

## 1. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 2. Treine o modelo

**Com dados reais:**
```bash
python src/train.py data/raw/DATASET_FIAP.csv
```

**Ou com dados sintéticos:**
```bash
python src/train.py
```

---

## 3. Suba a API

```bash
uvicorn api.app:app --reload
```
Acesse: http://127.0.0.1:8000

---

## 4. Teste o endpoint `/predict`

Exemplo com `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"INDE":7.5,"IDA":8.0,"IEG":7.0,"IAA":8.5,"IPS":7.0,"IPP":7.5,"IPV":0.8,"IAN":7.5,"FASE":2,"PEDRA":"Ametista"}'
```

Resposta esperada:
```json
{
  "prediction": 0,
  "label": "fora_do_grupo_de_risco",
  "drift_alerts": {}
}
```

---

## 5. Execute os testes unitários

```bash
pytest tests/
```

---

Pronto! Seu pipeline estará validado e a API pronta para uso.
