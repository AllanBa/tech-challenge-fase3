"""Train the baseline classifier, export ONNX, and compare inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from triage.data import read_dataset
from triage.predictor import OnnxPredictor


@dataclass(frozen=True)
class ModelConfiguration:
    name: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    min_df: int
    sublinear_tf: bool
    class_weight: str | None


MODEL_CONFIGURATION = ModelConfiguration(
    name="word_unigrams_balanced",
    analyzer="word",
    ngram_range=(1, 1),
    max_features=30_000,
    min_df=2,
    sublinear_tf=True,
    class_weight="balanced",
)


@dataclass(frozen=True)
class TrainingResult:
    samples: int
    classes: tuple[str, ...]
    model_configuration: str
    sklearn_median_latency_ms: float
    onnx_median_latency_ms: float
    latency_speedup: float
    evaluation: dict[str, object] | None


def evaluate_model(
    model: Pipeline,
    predictor: OnnxPredictor,
    data_path: Path,
    labels_path: Path | None,
) -> dict[str, object]:
    """Evaluate ONNX predictions against a labeled holdout CSV."""
    texts, expected = read_dataset(data_path, labels_path)
    classes = tuple(str(label) for label in model.classes_)
    unknown_labels = set(expected).difference(classes)
    if unknown_labels:
        raise ValueError(
            f"Evaluation data contains labels missing from training: "
            f"{sorted(unknown_labels)}"
        )

    sklearn_predictions = [str(label) for label in model.predict(texts)]
    onnx_predictions = [predictor.predict(text).label for text in texts]
    sklearn_probabilities = model.predict_proba(texts)
    disagreement_indexes = [
        index
        for index, (sklearn_label, onnx_label) in enumerate(
            zip(sklearn_predictions, onnx_predictions, strict=True)
        )
        if sklearn_label != onnx_label
    ]
    disagreement_margins = [
        float(np.diff(np.sort(sklearn_probabilities[index])[-2:])[0])
        for index in disagreement_indexes
    ]
    per_class = classification_report(
        expected,
        onnx_predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        expected,
        onnx_predictions,
        labels=classes,
    ).tolist()
    return {
        "samples": len(texts),
        "accuracy": round(accuracy_score(expected, onnx_predictions), 6),
        "macro_f1": round(
            f1_score(
                expected,
                onnx_predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            ),
            6,
        ),
        "weighted_f1": round(
            f1_score(
                expected,
                onnx_predictions,
                labels=classes,
                average="weighted",
                zero_division=0,
            ),
            6,
        ),
        "onnx_sklearn_agreement": round(
            accuracy_score(sklearn_predictions, onnx_predictions),
            6,
        ),
        "onnx_sklearn_disagreements": len(disagreement_indexes),
        "disagreement_sklearn_top2_margin_median": (
            round(float(np.median(disagreement_margins)), 6)
            if disagreement_margins
            else None
        ),
        "disagreement_sklearn_top2_margin_max": (
            round(max(disagreement_margins), 6) if disagreement_margins else None
        ),
        "class_report": per_class,
        "confusion_matrix": {
            "labels": list(classes),
            "rows": matrix,
        },
    }


def _median_latency_ms(
    function: Callable[[str], object],
    samples: list[str],
    repeats: int = 100,
) -> float:
    timings: list[float] = []
    for index in range(repeats):
        sample = samples[index % len(samples)]
        started = time.perf_counter()
        function(sample)
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def train_model(
    data_path: Path,
    artifact_dir: Path,
    labels_path: Path | None = None,
    test_data_path: Path | None = None,
) -> TrainingResult:
    """Fit a TF-IDF/logistic-regression model and export its ONNX equivalent."""
    if labels_path is None:
        default_labels_path = data_path.parent / "medical_tc_labels.csv"
        if default_labels_path.is_file():
            labels_path = default_labels_path
    texts, labels = read_dataset(data_path, labels_path)
    if len(set(labels)) < 2:
        raise ValueError("Training data must contain at least two distinct labels")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=MODEL_CONFIGURATION.analyzer,
                    ngram_range=MODEL_CONFIGURATION.ngram_range,
                    max_features=MODEL_CONFIGURATION.max_features,
                    min_df=MODEL_CONFIGURATION.min_df,
                    sublinear_tf=MODEL_CONFIGURATION.sublinear_tf,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=42,
                    class_weight=MODEL_CONFIGURATION.class_weight,
                ),
            ),
        ]
    )
    model.fit(texts, labels)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    sklearn_path = artifact_dir / "model.joblib"
    onnx_path = artifact_dir / "model.onnx"
    joblib.dump(model, sklearn_path)

    onnx_model = convert_sklearn(
        model,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={id(model.named_steps["classifier"]): {"zipmap": False}},
        target_opset=17,
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())

    classes = tuple(str(label) for label in model.classes_)
    predictor = OnnxPredictor(onnx_path, classes)
    if test_data_path is None:
        default_test_path = data_path.parent / "medical_tc_test.csv"
        if default_test_path.is_file():
            test_data_path = default_test_path
    evaluation = None
    if test_data_path is not None:
        test_labels_path = labels_path
        if test_labels_path is None:
            default_labels_path = test_data_path.parent / "medical_tc_labels.csv"
            if default_labels_path.is_file():
                test_labels_path = default_labels_path
        evaluation = evaluate_model(
            model,
            predictor,
            test_data_path,
            test_labels_path,
        )

    benchmark_texts = texts[: min(10, len(texts))]
    sklearn_latency = _median_latency_ms(
        lambda text: model.predict_proba([text]),
        benchmark_texts,
    )
    onnx_latency = _median_latency_ms(predictor.predict, benchmark_texts)
    speedup = sklearn_latency / onnx_latency if onnx_latency else 0.0
    result = TrainingResult(
        samples=len(texts),
        classes=classes,
        model_configuration=MODEL_CONFIGURATION.name,
        sklearn_median_latency_ms=round(sklearn_latency, 4),
        onnx_median_latency_ms=round(onnx_latency, 4),
        latency_speedup=round(speedup, 4),
        evaluation=evaluation,
    )
    (artifact_dir / "training_metrics.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/medical_abstracts/medical_tc_train.csv"),
        help="CSV with medical_abstract/condition_label or text/label columns.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        help="Optional CSV mapping condition_label to condition_name.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help=(
            "Optional labeled holdout CSV. Defaults to medical_tc_test.csv "
            "beside the training CSV when present."
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    result = train_model(
        args.data_path,
        args.artifact_dir,
        args.labels_path,
        args.test_data,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
