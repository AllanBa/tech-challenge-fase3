"""FastAPI application exposing the demonstration classifier and metrics."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

from triage.predictor import OnnxPredictor, Prediction

REQUEST_COUNT = Counter(
    "text_classifier_http_requests_total",
    "HTTP requests handled by the medical-abstract classification API.",
    ("method", "path", "status"),
)
REQUEST_LATENCY = Histogram(
    "text_classifier_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "path"),
)


class Predictor(Protocol):
    def predict(self, text: str) -> Prediction: ...


class ReportRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class ReportResponse(BaseModel):
    classification: str
    confidence: float
    notice: str = (
        "Educational topic classification only; not clinical advice or urgency triage."
    )


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if application.state.predictor is None:
            model_path = Path(os.getenv("MODEL_PATH", "artifacts/model.onnx"))
            metadata_path = model_path.parent / "training_metrics.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Model metadata does not exist: {metadata_path}. "
                    "Run `python -m triage.training` before starting the API."
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            application.state.predictor = OnnxPredictor(
                model_path,
                tuple(metadata["classes"]),
            )
        yield

    application = FastAPI(
        title="Medical Abstract Topic Classifier",
        description=(
            "Educational classification of medical abstracts into broad disease "
            "categories. It does not predict urgency and is not clinical advice."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.predictor = predictor

    @application.middleware("http")
    async def instrument_requests(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)
        started = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - started
            REQUEST_COUNT.labels(request.method, request.url.path, status).inc()
            REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "purpose": "medical abstract topic classification"}

    @application.post("/predict", response_model=ReportResponse)
    def predict(payload: ReportRequest, request: Request) -> ReportResponse:
        active_predictor: Predictor | None = request.app.state.predictor
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        result = active_predictor.predict(payload.text)
        return ReportResponse(
            classification=result.label,
            confidence=result.confidence,
        )

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
