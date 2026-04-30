#!/usr/bin/env python3
"""Export production artifacts for the ECG200 deployment kit.

Usage from the GitHub project root after training:

    python scripts/export_production_artifacts.py \
        --model results/models/rnn_seed0.keras \
        --out-dir ecg_mise_en_prod/models

If --model is omitted, the script reads results/metrics.csv and selects the row
with the best f1_macro, then accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TRAIN_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv"
TEST_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ECG200 model and preprocessing artifacts for production.")
    parser.add_argument("--model", type=Path, default=None, help="Path to the selected .keras model.")
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"), help="metrics.csv used when --model is omitted.")
    parser.add_argument("--out-dir", type=Path, default=Path("models"), help="Production models directory.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset cache directory.")
    parser.add_argument("--seed", type=int, default=None, help="Validation split seed used to train the selected model.")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--positive-label", default=None, help="Optional raw class value to display as abnormal/positive.")
    parser.add_argument("--negative-label", default=None, help="Optional raw class value to display as normal/negative.")
    return parser.parse_args()


def download_ecg200(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "ECG200_TRAIN.tsv"
    test_path = data_dir / "ECG200_TEST.tsv"
    if not train_path.exists():
        urllib.request.urlretrieve(TRAIN_URL, train_path)
    if not test_path.exists():
        urllib.request.urlretrieve(TEST_URL, test_path)
    return train_path, test_path


def read_ecg_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t", header=None).dropna()
    labels = df.iloc[:, 0].to_numpy()
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return features, labels


def select_best_model(metrics_path: Path) -> tuple[Path, int, dict[str, str]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metrics file is empty: {metrics_path}")

    def score(row: dict[str, str]) -> tuple[float, float]:
        return (float(row.get("f1_macro") or 0.0), float(row.get("accuracy") or 0.0))

    best = max(rows, key=score)
    model_name = best["model"]
    seed = int(float(best["seed"]))
    model_path = metrics_path.parent / "models" / f"{model_name}_seed{seed}.keras"
    return model_path, seed, best


def infer_seed_from_filename(model_path: Path) -> int | None:
    match = re.search(r"_seed(\d+)", model_path.stem)
    return int(match.group(1)) if match else None


def build_preprocess_config(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    train_path, _ = download_ecg200(args.data_dir)
    x_train_full, y_train_full_raw = read_ecg_tsv(train_path)

    label_encoder = LabelEncoder()
    y_train_full_labels = label_encoder.fit_transform(y_train_full_raw)

    x_train_raw, _, _, _ = train_test_split(
        x_train_full,
        y_train_full_labels,
        test_size=args.validation_size,
        random_state=seed,
        stratify=y_train_full_labels,
    )

    scaler = StandardScaler()
    scaler.fit(x_train_raw)

    class_names = [str(value) for value in label_encoder.classes_]
    display_labels: dict[str, str] = {}
    if args.positive_label is not None:
        display_labels[str(args.positive_label)] = "abnormal"
    if args.negative_label is not None:
        display_labels[str(args.negative_label)] = "normal"

    return {
        "dataset": "ECG200",
        "input_length": int(x_train_full.shape[1]),
        "validation_size": args.validation_size,
        "seed": seed,
        "class_names": class_names,
        "display_labels": display_labels,
        "scaler": {
            "type": "sklearn.preprocessing.StandardScaler",
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "var": scaler.var_.astype(float).tolist(),
            "n_features_in": int(scaler.n_features_in_),
        },
    }


def main() -> None:
    args = parse_args()

    if args.model is None:
        model_path, selected_seed, best_row = select_best_model(args.metrics)
        print(f"Selected best model from metrics: {best_row}")
    else:
        model_path = args.model
        selected_seed = args.seed if args.seed is not None else infer_seed_from_filename(model_path)

    if selected_seed is None:
        raise ValueError("Cannot determine seed. Pass --seed with the same seed used for training.")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, args.out_dir / "ecg_model.keras")
    preprocess_config = build_preprocess_config(args, selected_seed)
    with (args.out_dir / "preprocess.json").open("w", encoding="utf-8") as handle:
        json.dump(preprocess_config, handle, indent=2)
        handle.write("\n")

    print(f"Copied model to {args.out_dir / 'ecg_model.keras'}")
    print(f"Wrote preprocessing config to {args.out_dir / 'preprocess.json'}")


if __name__ == "__main__":
    main()
