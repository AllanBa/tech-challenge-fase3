FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.onnx
WORKDIR /app

COPY pyproject.toml README.md ./
COPY triage ./triage
RUN python -m pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
