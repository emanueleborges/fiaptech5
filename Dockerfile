# Usar imagem base oficial do Python
FROM python:3.9-slim

# Definir diretorio de trabalho
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o codigo fonte
COPY . .

# Expor a porta da API
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
