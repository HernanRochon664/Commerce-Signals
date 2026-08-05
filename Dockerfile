# Minimal FastAPI serving image for Commerce Signals.
#
# M0/M1 - installs runtime dependencies only via `uv sync
# --no-install-project`. Does NOT build or install the
# commerce-signals package itself: app/main.py is currently
# standalone and does not import from src/commerce_signals, so there
# is no reason to build the local package inside this image yet.
#
# When M6 wires FastAPI to the trained model/pipeline and needs
# src/commerce_signals, this Dockerfile needs revisiting: hatchling
# requires pyproject.toml's `readme` field to point at a file that
# exists in the build context, so installing the local package will
# need README.md copied in too (or a restructured pyproject.toml).
#
# Hardened further in M6: non-root user, healthcheck, multi-stage build.
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
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
