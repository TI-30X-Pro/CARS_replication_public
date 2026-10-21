FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV TZ=Europe/London
ENV PIP_NO_CACHE_DIR=1

# System dependencies and Python 3.9
RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    curl \
    libxcb1 \
    libgl1 \
    ffmpeg \
 && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
 && ln -sf /usr/bin/python3.9 /usr/bin/python \
 && ln -sf /usr/local/bin/pip /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

# Copy local packages used during environment installation
COPY CTG /workspace/CTG
COPY trajdata /workspace/trajdata
COPY Pplan /workspace/Pplan

# PyTorch dependencies
RUN pip install \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    torchmetrics==0.11.1 \
    torchtext \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Standard third-party dependencies
RUN pip install \
    tianshou \
    numba==0.56.4 \
    wandb \
    pytorch-lightning==2.0 \
    opencv-python \
    zarr \
    prettytable \
    protobuf==3.20.1 \
    transforms3d \
    scipy \
    dill \
    seaborn \
    einops==0.6.0 \
    casadi \
    pyarrow \
    pymap3d \
    nuscenes-devkit==1.1.9 \
    imageio \
    scikit-learn==1.2.2

# Install selected packages without dependency resolution
RUN pip install l5kit --no-deps
RUN pip install kornia==0.6.4 --no-deps
RUN pip install shapely --no-deps
RUN pip install pyemd --no-deps

RUN pip install POT --no-deps

# Pin final numerical stack versions
RUN pip uninstall -y numpy pandas matplotlib || true \
 && pip install \
    numpy==1.21.5 \
    pandas==1.4.1 \
    matplotlib==3.5.0

# Install local packages without pulling setup.py dependencies
WORKDIR /workspace/CTG
RUN pip install -e . --no-deps

WORKDIR /workspace/trajdata
RUN pip install -e . --no-deps

WORKDIR /workspace/Pplan
RUN pip install -e . --no-deps

WORKDIR /workspace
