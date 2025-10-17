FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash ca-certificates curl wget git \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MAMBA_ROOT_PREFIX=/opt/conda
SHELL ["/bin/bash", "-lc"]
RUN curl -L -o /tmp/micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest && \
    mkdir -p /opt/micromamba && \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /opt/micromamba --strip-components=1 && \
    ln -s /opt/micromamba/bin/micromamba /usr/local/bin/micromamba

WORKDIR /app
COPY . /app

RUN micromamba create -y -n meshsegnet -f envs/meshsegnet.yml && \
    micromamba create -y -n pointnetreg -f envs/pointnetreg.yml && \
    micromamba create -y -n calc -f envs/calc.yml && \
    micromamba clean -a -y

RUN micromamba run -n calc pip install -r requirements.api.txt

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD micromamba run -n calc uvicorn api:app --host 0.0.0.0 --port 8000
