"""ONNX Runtime inference for the optimized text classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class OnnxPredictor:
    def __init__(self, model_path: Path, classes: tuple[str, ...]) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        if not classes:
            raise ValueError("At least one model class is required")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._classes = classes

    def predict(self, text: str) -> Prediction:
        outputs = self._session.run(
            None,
            {self._input_name: np.asarray([[text]], dtype=object)},
        )
        labels = outputs[0]
        probabilities = outputs[1]
        label_value = labels[0]
        if isinstance(label_value, bytes):
            label_value = label_value.decode("utf-8")
        label = str(label_value)
        try:
            label_index = self._classes.index(label)
        except ValueError as error:
            raise RuntimeError(
                f"ONNX model returned an unknown label: {label}"
            ) from error
        confidence = float(np.asarray(probabilities)[0][label_index])
        return Prediction(label=label, confidence=confidence)
