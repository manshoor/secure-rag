FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Dependencies first, so code changes do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/docs ./data/docs
COPY data/golden.json ./data/golden.json
RUN uv sync --frozen --no-dev

EXPOSE 8010

# setup only adds missing users and documents, so restarts keep live access changes.
CMD ["sh", "-c", "rag db setup && uvicorn secure_rag.api:app --host 0.0.0.0 --port 8010"]
