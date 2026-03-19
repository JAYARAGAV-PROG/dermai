FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    timm==0.9.16 \
    Pillow==10.3.0 \
    numpy==1.26.4 \
    pydantic==2.7.0

# Copy code and model
COPY server.py .
COPY models/ models/

EXPOSE 8000

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
