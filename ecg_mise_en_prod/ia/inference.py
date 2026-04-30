from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class Prediction:
    predicted_class_index: int
    predicted_class_name: str
    predicted_display_label: str
    confidence: float
    probability_by_class: Dict[str, float]
    raw_output: List[float]


class ECGClassifier:
    """Inference wrapper for the ECG200 models trained in the GitHub project.

    The training code normalizes ECG200 with sklearn StandardScaler fitted on the
    train split only. In production, the same mean/scale values must be provided
    through /app/models/preprocess.json.
    """

    def __init__(self) -> None:
        self.model_path = Path(os.getenv("MODEL_PATH", "/app/models/ecg_model.keras"))
        self.preprocess_path = Path(os.getenv("PREPROCESS_PATH", "/app/models/preprocess.json"))
        self.preprocess_mode = os.getenv("PREPROCESS_MODE", "standard_scaler")
        self.input_length = int(os.getenv("INPUT_LENGTH", "96"))
        self.model: tf.keras.Model | None = None
        self.input_shape: Tuple[Any, ...] | None = None
        self.preprocess_config: Dict[str, Any] = {}
        self.class_names: List[str] = ["0", "1"]
        self.display_labels: Dict[str, str] = {}
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "model_loaded": self.model is not None,
            "model_input_shape": self.input_shape,
            "preprocess_path": str(self.preprocess_path),
            "preprocess_exists": self.preprocess_path.exists(),
            "preprocess_mode": self.preprocess_mode,
            "input_length": self.input_length,
            "class_names": self.class_names,
            "display_labels": self.display_labels,
        }

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Copy the selected .keras model to models/ecg_model.keras."
            )
        self._load_preprocess_config()
        self.model = tf.keras.models.load_model(self.model_path)
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.input_shape = tuple(shape)
        self._warmup()

    def _load_preprocess_config(self) -> None:
        if self.preprocess_path.exists():
            with self.preprocess_path.open("r", encoding="utf-8") as handle:
                self.preprocess_config = json.load(handle)
        else:
            self.preprocess_config = {}

        self.input_length = int(self.preprocess_config.get("input_length", self.input_length))
        self.class_names = [str(v) for v in self.preprocess_config.get("class_names", self.class_names)]
        self.display_labels = {
            str(k): str(v) for k, v in self.preprocess_config.get("display_labels", {}).items()
        }

        scaler = self.preprocess_config.get("scaler") or {}
        if "mean" in scaler and "scale" in scaler:
            self.scaler_mean = np.asarray(scaler["mean"], dtype=np.float32)
            self.scaler_scale = np.asarray(scaler["scale"], dtype=np.float32)
            if self.scaler_mean.size != self.input_length or self.scaler_scale.size != self.input_length:
                raise ValueError(
                    "preprocess.json scaler size does not match input_length="
                    f"{self.input_length}. Got mean={self.scaler_mean.size}, scale={self.scaler_scale.size}."
                )
        elif self.preprocess_mode == "standard_scaler":
            raise FileNotFoundError(
                "PREPROCESS_MODE=standard_scaler requires models/preprocess.json. "
                "Generate it with scripts/export_production_artifacts.py."
            )

    def _warmup(self) -> None:
        dummy = np.zeros((self.input_length,), dtype=np.float32)
        self.predict_one(dummy.tolist())

    def _preprocess_signal(self, values: Sequence[float]) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        if x.size != self.input_length:
            raise ValueError(f"Expected {self.input_length} ECG values, got {x.size}")

        if self.preprocess_mode == "standard_scaler":
            if self.scaler_mean is None or self.scaler_scale is None:
                raise RuntimeError("StandardScaler statistics are not loaded")
            scale = np.where(np.abs(self.scaler_scale) > 1e-12, self.scaler_scale, 1.0)
            x = (x - self.scaler_mean) / scale
        elif self.preprocess_mode == "per_sample_zscore":
            std = float(np.std(x))
            mean = float(np.mean(x))
            x = (x - mean) / std if std > 1e-12 else x - mean
        elif self.preprocess_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported PREPROCESS_MODE={self.preprocess_mode}")

        return x.astype(np.float32)

    def _shape_for_model(self, x: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise RuntimeError("Model is not loaded")
        rank = len(self.input_shape)
        if rank == 2:
            return x.reshape(1, self.input_length)
        if rank == 3:
            return x.reshape(1, self.input_length, 1)
        raise ValueError(f"Unsupported model input shape: {self.input_shape}")

    def _as_probabilities(self, raw: np.ndarray) -> tuple[np.ndarray, list[float]]:
        y = np.asarray(raw).reshape(-1).astype(np.float64)
        raw_list = [float(v) for v in y.tolist()]

        if y.size == 1:
            p1 = float(y[0])
            if p1 < 0.0 or p1 > 1.0:
                p1 = float(1.0 / (1.0 + np.exp(-p1)))
            return np.asarray([1.0 - p1, p1], dtype=np.float64), raw_list

        if np.any(y < 0.0) or not np.isclose(float(np.sum(y)), 1.0, atol=1e-3):
            shifted = y - np.max(y)
            exp = np.exp(shifted)
            y = exp / np.sum(exp)
        return y, raw_list

    def predict_one(self, signal: Sequence[float]) -> Prediction:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        x = self._preprocess_signal(signal)
        batch = self._shape_for_model(x)
        raw = self.model.predict(batch, verbose=0)
        probabilities, raw_list = self._as_probabilities(raw)

        predicted_index = int(np.argmax(probabilities))
        if predicted_index < len(self.class_names):
            class_name = self.class_names[predicted_index]
        else:
            class_name = str(predicted_index)
        display_label = self.display_labels.get(class_name, class_name)

        probability_by_class = {}
        for index, probability in enumerate(probabilities.tolist()):
            name = self.class_names[index] if index < len(self.class_names) else str(index)
            probability_by_class[name] = float(probability)

        return Prediction(
            predicted_class_index=predicted_index,
            predicted_class_name=class_name,
            predicted_display_label=display_label,
            confidence=float(probabilities[predicted_index]),
            probability_by_class=probability_by_class,
            raw_output=raw_list,
        )

    def predict_many(self, signals: Sequence[Sequence[float]]) -> List[Prediction]:
        return [self.predict_one(signal) for signal in signals]
