# CPU / smoke image (Python 3.12). For GPU: ./scripts/docker.sh build-gpu
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY adapters ./adapters
COPY bench ./bench
COPY streaming ./streaming
COPY scripts ./scripts
COPY configs ./configs
COPY fixtures ./fixtures

# CPU torch first so pip does not pull multi-GB CUDA wheels into the smoke image.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -e .

RUN mkdir -p /app/data /app/results /app/.cache

EXPOSE 8000

CMD ["stream", "--host", "0.0.0.0", "--port", "8000"]
