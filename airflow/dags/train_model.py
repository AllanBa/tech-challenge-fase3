"""Airflow DAG for validating the corpus CSV and retraining the classifier."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from airflow.decorators import dag, task

from triage.data import read_dataset
from triage.training import train_model


def project_root() -> Path:
    configured_root = os.environ.get("TRIAGE_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root)
    return Path(__file__).resolve().parents[2]


@dag(
    dag_id="medical_abstract_topic_training",
    description="Validate the medical abstracts CSV, train, and export the ONNX model.",
    schedule=None,
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["demo", "training"],
)
def medical_abstract_topic_training():
    @task
    def validate_training_csv() -> str:
        default_path = (
            project_root()
            / "data"
            / "medical_abstracts"
            / "medical_tc_train.csv"
        )
        configured_path = os.environ.get("TRIAGE_DATA_PATH")
        csv_path = Path(configured_path) if configured_path else default_path
        if not csv_path.is_absolute():
            csv_path = project_root() / csv_path
        texts, _ = read_dataset(csv_path)
        print(f"Validated {len(texts)} training rows from {csv_path}")
        return str(csv_path)

    @task
    def train_and_export(csv_path: str) -> dict[str, object]:
        result = train_model(Path(csv_path), project_root() / "artifacts")
        return {
            "samples": result.samples,
            "classes": list(result.classes),
            "sklearn_median_latency_ms": result.sklearn_median_latency_ms,
            "onnx_median_latency_ms": result.onnx_median_latency_ms,
            "latency_speedup": result.latency_speedup,
            "evaluation": result.evaluation or {},
        }

    train_and_export(validate_training_csv())


medical_abstract_topic_training()
