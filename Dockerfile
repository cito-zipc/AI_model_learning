FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .project-root pyproject.toml ./

ENV PYTHONPATH=/app \
    PROJECT_ROOT=/app

CMD ["python", "src/train.py"]
