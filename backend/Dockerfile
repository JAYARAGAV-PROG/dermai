FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch (same accuracy as GPU, just runs inference slightly slower)
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

# Copy code and trained model
COPY server.py .
COPY models/ models/

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
