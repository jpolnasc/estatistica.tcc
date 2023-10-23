# Usamos uma imagem base que já contém R e Python
FROM quay.io/jupyter/datascience-notebook:latest

# Atualizamos a lista de pacotes e instalamos o software necessário
USER root
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev

# Mudamos para o usuário jovyan (o usuário padrão nesta imagem)
USER jovyan

# Criamos um ambiente virtual
RUN python3 -m venv myenv

# Adicionamos a ativação do ambiente virtual ao ~/.bashrc,
# desativando primeiro o Conda, se estiver ativo
RUN echo "if ! [ -z \"\$CONDA_PREFIX\" ]; then conda deactivate; fi" >> ~/.bashrc && \
    echo "source ../../home/jovyan/myenv/bin/activate" >> ~/.bashrc

# Ativamos o ambiente virtual e instalamos as bibliotecas de Python
RUN /bin/bash -c "source myenv/bin/activate && \
    pip install --no-cache-dir \
    scikit-learn \
    mlflow \
    river \
    pandas \
    matplotlib \
    seaborn"

# Instalamos as bibliotecas de R
RUN R -e "install.packages(c('tidyverse', 'simts', 'languageserver', 'forecast'), repos='https://cloud.r-project.org/')"
