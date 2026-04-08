# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Code Catalyst"
LABEL description="Email Triage OpenEnv Environment"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (cached layer) ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY main.py              .
COPY models.py            .
COPY inference.py         .
COPY triage_grader_v2.py  .
COPY openenv.yaml         .
COPY data/                ./data/

# ── Environment variables (overridden at runtime) ─────────────────────────────
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV HF_TOKEN=""
ENV ENV_URL="http://localhost:7860"

# ── Expose port (HF Spaces requires 7860) ────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start the FastAPI server ──────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
