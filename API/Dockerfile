ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV MAMBA_ROOT_PREFIX=/opt/conda
SHELL ["/bin/bash", "-lc"]

COPY build-tools/micromamba-linux /usr/local/bin/micromamba
RUN chmod +x /usr/local/bin/micromamba

# Configure faster conda/pip mirrors
RUN mkdir -p /root/.config/pip
COPY .condarc /root/.condarc
COPY pip.conf /root/.config/pip/pip.conf

WORKDIR /app
COPY . /app
RUN chmod +x /app/start.sh /app/run.sh

RUN micromamba create -y -n ios_model -f envs/ios_model.yml && \
    micromamba create -y -n pointnetreg -f envs/pointnetreg.yml && \
    micromamba create -y -n calc -f envs/calc.yml && \
    micromamba clean -a -y

RUN micromamba run -n ios_model pip install --no-cache-dir \
        torch_scatter \
        torch_sparse \
        torch_cluster \
        torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && \
    micromamba run -n ios_model pip install --no-cache-dir torch_geometric==2.5.3 && \
    micromamba clean -a -y

ENV PYTHONUNBUFFERED=1 \
    MICROMAMBA_EXE=/usr/local/bin/micromamba \
    APP_VERSION=v1.0.0

EXPOSE 12499

ENTRYPOINT ["/usr/local/bin/micromamba", "run", "-n", "calc", "uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "12499"]
