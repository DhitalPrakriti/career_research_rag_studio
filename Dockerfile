# Two stages: build the frontend with node, then serve everything from one Python image.
# FastAPI serves the built assets, so the deployed unit is a single container and the
# browser never makes a cross-origin request.

FROM node:22-slim AS frontend
WORKDIR /build

# Dependencies first, so a source-only change does not reinstall node_modules.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit

COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the model cache inside the image so a cold start does not download it.
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch, installed before the project so the default CUDA build is never pulled.
# The GPU wheels are several gigabytes and useless on Cloud Run.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.9.1"

# Install sentence-transformers and bake the embedding model in BEFORE copying source, so
# editing the app does not re-download 90 MB on every rebuild. Without baking it at all,
# every cold start would fetch the model before answering, at the mercy of HF rate limits.
# The offline flags are deliberately set *after* the download: setting them earlier makes
# the download fail against its own cache policy, which is what broke the first build.
RUN pip install --no-cache-dir "sentence-transformers==5.4.1"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && find /opt/hf \( -name '*.h5' -o -name '*.ot' -o -name 'tf_model*' \) -delete

# Cache is populated, so forbid runtime downloads: a cold start must not depend on Hugging
# Face being reachable.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY pyproject.toml README.md ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir ".[api,agent]" "python-multipart>=0.0.9"

COPY --from=frontend /build/dist ./frontend/dist

# Seed documents. On Cloud Run this directory is replaced by a mounted GCS bucket so
# uploads survive restarts; locally it is just the folder on disk.
COPY docs/ ./docs/

# Run unprivileged. The documents directory stays writable so uploads work.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/docs \
    && chown -R appuser:appuser /app/docs
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os; \
urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','8080')}/healthz\").read()"

# sh -c so $PORT expands: Cloud Run assigns the port and expects the app to honour it.
# One worker on purpose — each worker would hold its own copy of the model and index.
CMD ["sh", "-c", "exec uvicorn --factory rag_studio.api.app:create_app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
