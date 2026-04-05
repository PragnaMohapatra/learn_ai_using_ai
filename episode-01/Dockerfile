# ============================================================
# Decode AI Using AI — Episode 1
# Python CLI: Data Cleaner + Notebook Runner
# ============================================================

# Start from official Python 3.12 (slim = smaller image, no extras)
FROM python:3.12-slim

# Labels (metadata, not functional)
LABEL maintainer="Decode AI Using AI"
LABEL description="Episode 1: Data cleaning and notebook automation CLI"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /workspace
ENV PYTHONPATH=/workspace

# Install dependencies first (layer caching — explained below)
COPY episode-01/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the IPython kernel so notebooks can execute
RUN python -m ipykernel install --user --name python3

# Copy shared and episode-specific source code
COPY shared/ ./shared/
COPY episode-01/src/ ./src/

# Copy sample data and notebooks
COPY episode-01/data/ ./data/
COPY episode-01/notebooks/ ./notebooks/

# Default entrypoint: the CLI
ENTRYPOINT ["python", "-m", "src.cli"]

# Default command: show help
CMD ["--help"]
