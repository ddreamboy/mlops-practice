FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

COPY pyproject.toml .
RUN uv pip install --no-cache-dir --system .

RUN groupadd -r appuser && useradd -r -g appuser appuser && chown appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]
