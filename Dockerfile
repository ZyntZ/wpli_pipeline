FROM python:3.11-slim

WORKDIR /work
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /work/
COPY src /work/src
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["wpli-run"]
