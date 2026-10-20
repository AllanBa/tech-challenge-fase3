from fastapi.testclient import TestClient

from triage.api import create_app
from triage.predictor import Prediction


class FakePredictor:
    def predict(self, text: str) -> Prediction:
        assert text == "fictional medical abstract"
        return Prediction(label="Neoplasms", confidence=0.93)


def test_health_prediction_and_metrics():
    client = TestClient(create_app(FakePredictor()))

    health = client.get("/health")
    prediction = client.post("/predict", json={"text": "fictional medical abstract"})
    metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["purpose"] == "medical abstract topic classification"
    assert prediction.status_code == 200
    assert prediction.json()["classification"] == "Neoplasms"
    assert prediction.json()["confidence"] == 0.93
    assert "not clinical advice or urgency triage" in prediction.json()["notice"]
    assert metrics.status_code == 200
    assert "text_classifier_http_requests_total" in metrics.text
    assert "text_classifier_http_request_duration_seconds" in metrics.text


def test_blank_report_is_rejected():
    client = TestClient(create_app(FakePredictor()))

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
