FROM python:3.10

# Install system dependencies for PyBullet, graphics, and X11
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    xvfb \
    mesa-utils \
    libgl1-mesa-dri \
    libglx-mesa0 \
    x11-apps \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Set up display (can use X11 or Xvfb)
ENV DISPLAY=:99

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY SamuraiProject/requirements.txt /workspace/requirements.txt

# Install PyTorch with CUDA support from PyTorch repo
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY SamuraiProject/ /workspace/SamuraiProject/
COPY samurai_rl/ /workspace/samurai_rl/

# Set Python path
ENV PYTHONPATH=/workspace:/workspace/SamuraiProject:${PYTHONPATH}

# Create startup script that can handle both X11 and Xvfb
RUN echo '#!/bin/bash\n\
# Try X11 first, fall back to Xvfb if not available\n\
if [ -n "$DISPLAY" ] && xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then\n\
    echo "Using X11 display: $DISPLAY"\n\
else\n\
    echo "Starting Xvfb for headless rendering"\n\
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
    export DISPLAY=:99\n\
    sleep 2\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "SamuraiProject/train_samurai.py"]