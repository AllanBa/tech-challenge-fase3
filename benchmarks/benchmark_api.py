"""Measure end-to-end latency of the running classification API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from triage.data import read_dataset


def percentile(values: list[float], quantile: float) -> float:
    """Calculate a nearest-rank percentile from non-empty millisecond values."""
    if not values:
        raise ValueError("At least one latency measurement is required")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in the interval (0, 1]")
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[index]


def summarize_latencies(
    latencies_ms: list[float],
    elapsed_seconds: float,
) -> dict[str, float]:
    """Return end-to-end percentiles and throughput for completed requests."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return {
        "p50_ms": round(percentile(latencies_ms, 0.50), 4),
        "p95_ms": round(percentile(latencies_ms, 0.95), 4),
        "p99_ms": round(percentile(latencies_ms, 0.99), 4),
        "mean_ms": round(statistics.fmean(latencies_ms), 4),
        "throughput_requests_per_second": round(
            len(latencies_ms) / elapsed_seconds,
            4,
        ),
        "elapsed_seconds": round(elapsed_seconds, 4),
    }


async def run_benchmark(
    url: str,
    text: str,
    *,
    requests: int,
    concurrency: int,
    warmup: int,
    timeout_seconds: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Warm the API, run concurrent requests, and return a reproducible summary."""
    if requests < 1:
        raise ValueError("requests must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if not text.strip():
        raise ValueError("text must not be blank")

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(timeout_seconds)
    semaphore = asyncio.Semaphore(concurrency)
    measured_latencies: list[float] = []
    classifications: Counter[str] = Counter()

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        transport=transport,
    ) as client:
        health = await client.get(f"{url.rstrip('/')}/health")
        health.raise_for_status()

        for _ in range(warmup):
            response = await client.post(
                f"{url.rstrip('/')}/predict",
                json={"text": text},
            )
            response.raise_for_status()

        async def send_request() -> tuple[float, str]:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    f"{url.rstrip('/')}/predict",
                    json={"text": text},
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                payload = response.json()
                classification = payload.get("classification")
                confidence = payload.get("confidence")
                if not isinstance(classification, str) or not isinstance(
                    confidence, (int, float)
                ):
                    raise ValueError(
                        "Prediction response must include classification and confidence"
                    )
                return elapsed_ms, classification

        started = time.perf_counter()
        results = await asyncio.gather(
            *(send_request() for _ in range(requests))
        )
        elapsed_seconds = time.perf_counter() - started

    for latency, classification in results:
        measured_latencies.append(latency)
        classifications[classification] += 1

    summary = summarize_latencies(measured_latencies, elapsed_seconds)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "endpoint": f"{url.rstrip('/')}/predict",
        "requests": requests,
        "warmup_requests": warmup,
        "concurrency": concurrency,
        "request_text_characters": len(text),
        "status": "all_requests_succeeded",
        "classification_counts": dict(sorted(classifications.items())),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/medical_abstracts/medical_tc_test.csv"),
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based row index used as the repeated benchmark input.",
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    texts, _ = read_dataset(args.data_path)
    if args.sample_index >= len(texts):
        parser.error(
            f"--sample-index must be less than the {len(texts)} available rows"
        )

    result = asyncio.run(
        run_benchmark(
            args.url,
            texts[args.sample_index],
            requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
            timeout_seconds=args.timeout,
        )
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
