# Minimal FastAPI serving image for Commerce Signals.
#
# M0 placeholder — will be hardened in M6 with non-root user, health
# checks, multi-stage model loading, etc.
#
# Build:
#     docker build -t commerce-signals:latest .
#
# Run:
#     docker run --rm -p 8000:8000 commerce-signals:latest

FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .
COPY src/ src/
COPY app/ app/

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]