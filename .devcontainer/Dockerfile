# Usamos uma imagem base que já contém R e Python
FROM quay.io/jupyter/datascience-notebook:latest

# Atualizamos a lista de pacotes e instalamos o software necessário
USER root
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    telnet

# Mudamos para o usuário jovyan (o usuário padrão nesta imagem)
USER jovyan

RUN conda install -c conda-forge \
    jupyter \
    scikit-learn \
    mlflow \
    pandas \
    matplotlib \
    seaborn \
    pmdarima \
    psycopg2 -y && \
    pip install river

RUN echo "if ! [ -z \"\$CONDA_PREFIX\" ]; then conda deactivate; fi" >> ~/.bashrc

# Instalamos as bibliotecas de R
RUN R -e "install.packages(c('tidyverse', 'simts', 'languageserver', 'forecast'), repos='https://cloud.r-project.org/')"
