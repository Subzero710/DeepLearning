from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from flask import Flask, jsonify, request

from inference import ECGClassifier

app = Flask(__name__)
classifier = ECGClassifier()


def error_response(message: str, status: int = 400):
    return jsonify({"status": "error", "error": message, "metadata": classifier.metadata()}), status


def ensure_loaded():
    if classifier.model is not None:
        return None
    try:
        classifier.load()
    except Exception as exc:
        return error_response(str(exc), 503)
    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "metadata": classifier.metadata()})


@app.route("/config", methods=["GET", "POST"])
def config():
    load_error = ensure_loaded()
    if load_error is not None:
        return load_error
    return jsonify({"status": "ok", "metadata": classifier.metadata()})


@app.route("/predict", methods=["POST"])
def predict():
    load_error = ensure_loaded()
    if load_error is not None:
        return load_error
    payload: Dict[str, Any] | None = request.get_json(silent=True)
    if not payload or "signal" not in payload:
        return error_response('Expected JSON payload: {"signal": [96 numeric values]}')
    try:
        prediction = classifier.predict_one(payload["signal"])
    except Exception as exc:
        return error_response(str(exc))
    return jsonify({"status": "ok", "data": asdict(prediction), "metadata": classifier.metadata()})


@app.route("/batch", methods=["POST"])
def batch():
    load_error = ensure_loaded()
    if load_error is not None:
        return load_error
    payload: Dict[str, Any] | None = request.get_json(silent=True)
    if not payload or "signals" not in payload:
        return error_response('Expected JSON payload: {"signals": [[96 numeric values], ...]}')
    try:
        predictions = [asdict(p) for p in classifier.predict_many(payload["signals"])]
    except Exception as exc:
        return error_response(str(exc))
    return jsonify({"status": "ok", "data": predictions, "metadata": classifier.metadata()})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "routes": ["/health", "/config", "/predict", "/batch"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False, use_reloader=False)
