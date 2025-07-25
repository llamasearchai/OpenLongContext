# Multi-stage build for OpenLongContext
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Copy requirements and setup files
COPY --chown=builder:builder requirements.txt setup.py pyproject.toml ./
COPY --chown=builder:builder openlongcontext/ ./openlongcontext/

# Create virtual environment and install dependencies
RUN python -m venv /home/builder/venv
ENV PATH="/home/builder/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip install -e .[all] && \
    pip install gunicorn uvicorn[standard]

# Production stage
FROM python:3.13-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user
RUN useradd --create-home --shell /bin/bash --user-group openlongcontext
USER openlongcontext
WORKDIR /home/openlongcontext

# Copy virtual environment from builder
COPY --from=builder --chown=openlongcontext:openlongcontext /home/builder/venv ./venv

# Copy application code
COPY --chown=openlongcontext:openlongcontext openlongcontext/ ./openlongcontext/
COPY --chown=openlongcontext:openlongcontext configs/ ./configs/
COPY --chown=openlongcontext:openlongcontext docs/ ./docs/
COPY --chown=openlongcontext:openlongcontext OpenContext.png ./
COPY --chown=openlongcontext:openlongcontext README.md LICENSE ./

# Set environment variables
ENV PATH="/home/openlongcontext/venv/bin:$PATH"
ENV PYTHONPATH="/home/openlongcontext:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "openlongcontext.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER openlongcontext

# Install development Python packages
RUN pip install --upgrade pip && \
    pip install -e .[dev,research,agents,all]

# Override command for development
CMD ["uvicorn", "openlongcontext.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 