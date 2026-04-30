"""Shared utilities for ECG200 experiments."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and TensorFlow seeds."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_output_dirs(base_dir: str | Path = "results") -> dict[str, Path]:
    """Create output directories and return their paths."""

    root = Path(base_dir)
    paths = {
        "root": root,
        "training_curves": root / "training_curves",
        "confusion_matrices": root / "confusion_matrices",
        "models": root / "models",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def plot_training_curves(history: tf.keras.callbacks.History, output_path: str | Path) -> None:
    """Save train/validation loss and accuracy curves."""

    output_path = Path(output_path)
    history_dict = history.history

    plt.figure()
    plt.plot(history_dict.get("loss", []), label="train_loss")
    plt.plot(history_dict.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + "_loss.png"), dpi=150)
    plt.close()

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        plt.figure()
        plt.plot(history_dict["accuracy"], label="train_accuracy")
        plt.plot(history_dict["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path.with_name(output_path.stem + "_accuracy.png"), dpi=150)
        plt.close()


def save_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    """Save a confusion matrix as a CSV-like text file."""

    np.savetxt(output_path, matrix, fmt="%d", delimiter=",")


def model_file_size_mb(path: str | Path) -> float:
    """Return file size in MiB."""

    return Path(path).stat().st_size / (1024 * 1024)

def clear_previous_results(output_dirs: dict[str, Path]) -> None:
    """Delete generated result files from previous experiment runs."""

    files_to_delete = [
        output_dirs["root"] / "metrics.csv",
    ]

    patterns = {
        "confusion_matrices": ["*.csv"],
        "training_curves": ["*.png"],
        "models": ["*.keras", "*.h5"],
    }

    for directory_key, glob_patterns in patterns.items():
        directory = output_dirs[directory_key]
        for pattern in glob_patterns:
            files_to_delete.extend(directory.glob(pattern))

    for path in files_to_delete:
        if path.exists() and path.is_file():
            path.unlink()
