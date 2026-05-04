# =============================================================================
# Multi-stage Python Dockerfile Template
# =============================================================================

# =============================================================================
# Stage 1: Build dependencies and install package
# =============================================================================
FROM python:3.13-slim AS builder

# Build-time argument for version injection (e.g. via 'docker build --build-arg APP_VERSION=$(git describe)')
ARG APP_VERSION=dev

WORKDIR /app

# Copy uv from official image (faster and smaller than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Inject the build version into version.py before installing
RUN python -c "import pathlib; p=pathlib.Path('src/rag_tester/version.py'); p.write_text(f'__version__: str = \"${APP_VERSION}\"\n')"

# Install package and dependencies (without dev packages)
RUN uv sync --frozen --no-dev --no-editable

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user for security (UID 10001)
RUN groupadd --gid 10001 appgroup && \
    useradd \
        --uid 10001 \
        --gid appgroup \
        --shell /bin/false \
        --no-create-home \
        appuser

# Copy virtual environment from builder (includes installed package)
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Re-declare ARG in the runtime stage so we can promote it to ENV
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

# Switch to non-root user
USER appuser:appgroup

# CLI entry point exposed by [project.scripts] in pyproject.toml
ENTRYPOINT ["rag-tester"]
CMD ["--help"]
