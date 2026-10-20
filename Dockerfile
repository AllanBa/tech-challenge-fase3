FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.onnx \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y locales \
    && sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY triage ./triage
RUN python -m pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
