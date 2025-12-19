FROM python:3.10-slim

# ---------------- System libs for PyBullet + X11 ----------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    libx11-6 \
    libxrender1 \
    libsm6 \
    libxext6 \
    mesa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------- X11 / OpenGL env ----------------
ENV CUDA_VISIBLE_DEVICES=""
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# ---------------- Project files ----------------
WORKDIR /workspace
COPY . /workspace

WORKDIR /workspace/SamuraiProject

# ---------------- Python deps ----------------
RUN pip install --upgrade pip wheel setuptools

# CPU-only PyTorch
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime dependencies
RUN pip install -r requirements_runtime.txt

# ---------------- Default run ----------------
CMD ["python", "eval_samurai.py"]
