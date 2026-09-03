# FastAPI serving image for Commerce Signals (M6).
#
# M6 resolves the "needs revisiting" note from M0/M1: the API now imports
# from src/commerce_signals (features, explainability, datasets), so the
# image must make that package importable. Instead of installing the local
# package as editable (which would require hatchling/README.md handling that
# was deliberately avoided in M1), the source is copied and PYTHONPATH is
# set so src/ is on the import path. No `uv sync --no-install-project`
# change is needed beyond the copy + env var.
#
# Future hardening (M6 plan note): non-root user, healthcheck, multi-stage
# build — not required for M6 functional scope, but noted for later.
#
# Build:
#     docker build -t commerce-signals:latest .
#
# Run:
#     docker run --rm -p 8000:8000 commerce-signals:latest
FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY app/ app/
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
