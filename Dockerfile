FROM python:3.10-slim

# System libs for PyBullet + X11
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxrender1 \
    libsm6 \
    libxext6 \
    mesa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Force CPU (even if code says cuda)
ENV CUDA_VISIBLE_DEVICES=""

WORKDIR /workspace
COPY . /workspace

WORKDIR /workspace/SamuraiProject

RUN pip install --upgrade pip wheel setuptools

# CPU-only torch
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime deps only
RUN pip install -r requirements_runtime.txt

# Default run = visual demo
CMD ["python", "eval_samurai.py"]
