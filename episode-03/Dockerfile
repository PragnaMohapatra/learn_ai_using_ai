FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace:/workspace/episode-03

WORKDIR /workspace/episode-03

COPY episode-03/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY shared/ /workspace/shared/
COPY episode-03/ /workspace/episode-03/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]