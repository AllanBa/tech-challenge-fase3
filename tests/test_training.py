import csv
import json

from fastapi.testclient import TestClient

from triage.api import create_app
from triage.predictor import OnnxPredictor
from triage.training import train_model


def test_training_exports_onnx_and_latency_comparison(tmp_path, monkeypatch):
    data_path = tmp_path / "reports.csv"
    test_data_path = tmp_path / "test_reports.csv"
    labels_path = tmp_path / "medical_tc_labels.csv"
    artifact_dir = tmp_path / "artifacts"
    classes = {
        "1": ("Neoplasms", "fictional_neoplasm_marker"),
        "2": ("Digestive system diseases", "fictional_digestive_marker"),
        "3": ("Nervous system diseases", "fictional_nervous_marker"),
        "4": ("Cardiovascular diseases", "fictional_cardiovascular_marker"),
        "5": ("General pathological conditions", "fictional_general_marker"),
    }
    with data_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("medical_abstract", "condition_label"),
        )
        writer.writeheader()
        for label_id, (_, marker) in classes.items():
            for index in range(20):
                writer.writerow(
                    {
                        "medical_abstract": (
                            f"fictional abstract {marker} sample {index}"
                        ),
                        "condition_label": label_id,
                    }
                )
    with labels_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("condition_label", "condition_name"),
        )
        writer.writeheader()
        for label_id, (name, _) in classes.items():
            writer.writerow({"condition_label": label_id, "condition_name": name})

    with test_data_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("medical_abstract", "condition_label"),
        )
        writer.writeheader()
        for label_id, (_, marker) in classes.items():
            for index in range(2):
                writer.writerow(
                    {
                        "medical_abstract": (
                            f"heldout fictional abstract {marker} example {index}"
                        ),
                        "condition_label": label_id,
                    }
                )

    result = train_model(data_path, artifact_dir, labels_path, test_data_path)
    predictor = OnnxPredictor(
        artifact_dir / "model.onnx",
        result.classes,
    )
    prediction = predictor.predict(
        "fictional abstract fictional_cardiovascular_marker test"
    )
    metrics = json.loads((artifact_dir / "training_metrics.json").read_text())

    assert result.samples == 100
    assert result.model_configuration == "word_unigrams_balanced"
    assert set(result.classes) == {name for name, _ in classes.values()}
    assert prediction.label == "Cardiovascular diseases"
    assert 0 <= prediction.confidence <= 1
    assert (artifact_dir / "model.joblib").is_file()
    assert metrics["sklearn_median_latency_ms"] > 0
    assert metrics["onnx_median_latency_ms"] > 0
    assert result.evaluation is not None
    assert result.evaluation["samples"] == 10
    assert result.evaluation["accuracy"] == 1.0
    assert result.evaluation["onnx_sklearn_agreement"] == 1.0
    assert result.evaluation["onnx_sklearn_disagreements"] == 0
    assert result.evaluation["disagreement_sklearn_top2_margin_max"] is None

    monkeypatch.setenv("MODEL_PATH", str(artifact_dir / "model.onnx"))
    with TestClient(create_app()) as client:
        response = client.post(
            "/predict",
            json={
                "text": (
                    "fictional abstract fictional_cardiovascular_marker test"
                )
            },
        )

    assert response.status_code == 200
    assert response.json()["classification"] == "Cardiovascular diseases"
