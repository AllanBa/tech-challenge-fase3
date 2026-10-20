import asyncio
import json

import httpx
import pytest

from benchmarks.benchmark_api import percentile, run_benchmark, summarize_latencies


def test_percentile_uses_nearest_rank():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_percentile_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="At least one"):
        percentile([], 0.95)
    with pytest.raises(ValueError, match="quantile"):
        percentile([1.0], 0)


def test_summarize_latencies_reports_percentiles_and_throughput():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0], 2.0)

    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == 4.0
    assert summary["p99_ms"] == 4.0
    assert summary["throughput_requests_per_second"] == 2.0


def test_run_benchmark_warms_up_then_measures_concurrent_calls():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        payload = json.loads(request.content)
        assert payload == {"text": "fictional abstract"}
        return httpx.Response(
            200,
            json={"classification": "Neoplasms", "confidence": 0.9},
        )

    result = asyncio.run(
        run_benchmark(
            "http://benchmark.test",
            "fictional abstract",
            requests=8,
            concurrency=2,
            warmup=2,
            transport=httpx.MockTransport(handler),
        )
    )

    assert result["requests"] == 8
    assert result["warmup_requests"] == 2
    assert result["concurrency"] == 2
    assert result["status"] == "all_requests_succeeded"
    assert result["classification_counts"] == {"Neoplasms": 8}
    assert result["p95_ms"] >= result["p50_ms"]
