FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl unzip ca-certificates \
    build-essential patchelf \
    libosmesa6-dev libgl1-mesa-dev libglfw3 libglew-dev libegl1 \
    libxrender1 libxext6 libxtst6 libxrandr2 libxi6 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.9 python3.9-dev python3.9-venv python3.9-distutils && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.9 2

ARG MUJOCO_URL=https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN mkdir -p /root/.mujoco && cd /root/.mujoco && \
    wget -qO mujoco.tar.gz ${MUJOCO_URL} && tar -xzf mujoco.tar.gz && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_GL=egl

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "cython<3" wheel setuptools six && \
    python3 -m pip install mujoco-py==2.1.2.14 "gym==0.23.1" && \
    python3 -m pip install "torch>=1.12,<3.0" jax jaxlib flax optax && \
    python3 -m pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl" || true && \
    python3 -m pip install pygame einops && \
    python3 - <<'PY'
import sys, pkgutil
print("Python:", sys.version)
assert pkgutil.find_loader("gym"), "gym not installed"
assert pkgutil.find_loader("mujoco_py"), "mujoco_py not installed"
print("OK: gym & mujoco_py present")
PY

WORKDIR /workspace
# COPY . /workspace

CMD ["/bin/bash"]
