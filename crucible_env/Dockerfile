# HuggingFace Space: crucible-env
# SDK: docker | Port: 8000 | OpenEnv-compatible
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY . ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "openenv-core==0.2.3" \
       "huggingface_hub>=1.0.0" \
       "anthropic>=0.40.0" \
       "groq>=0.11.0" \
       "pydantic>=2.0.0" \
       "fastapi>=0.110.0" \
       "uvicorn[standard]>=0.29.0"

ENV PYTHONPATH=/app
ENV PORT=8000
ENV USE_ARCHITECT=false
ENV USE_ADVERSARIAL=false
ENV USE_SHOCKS=false
ENV JURISDICTION=FAR

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
