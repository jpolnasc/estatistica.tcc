# Usamos a imagem base oficial do Python
FROM python:3.8-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Atualiza a lista de pacotes e instala pacotes necessários
USER root
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala os pacotes Python especificados
RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    pmdarima \
    river

# Copiar o conteúdo da pasta scripts para dentro do container (ajuste conforme necessário)
COPY ./scripts /app/scripts

# Comando padrão para executar ao iniciar o container
CMD ["python", "./scripts/main.py"]