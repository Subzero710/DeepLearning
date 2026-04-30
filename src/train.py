"""Train and compare deep-learning models on ECG200."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import tensorflow as tf

from src.data import load_ecg200
from src.evaluate import evaluate_classifier
from src.models import get_model_builder
from src.utils import (
    clear_previous_results,
    ensure_output_dirs,
    model_file_size_mb,
    plot_training_curves,
    save_confusion_matrix,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG200 deep-learning models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "cnn"],
        choices=["mlp", "cnn", "rnn"],
        help="Models to train. RNN requires build_rnn() to be implemented.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Delete previous generated results before running the experiment.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping and train for the requested number of epochs.",
    )
    return parser.parse_args()


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def select_arrays(model_name: str, data):
    """Return arrays with the correct shape for the selected model."""

    if model_name == "mlp":
        return data.x_train_mlp, data.x_val_mlp, data.x_test_mlp, data.input_shape_mlp

    return data.x_train_seq, data.x_val_seq, data.x_test_seq, data.input_shape_seq


def train_one_model(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    output_dirs: dict[str, Path],
) -> dict[str, object]:
    """Train one model for one seed and return a metrics row."""

    set_global_seed(seed)

    data = load_ecg200(
        data_dir=args.data_dir,
        validation_size=args.validation_size,
        seed=seed,
    )

    x_train, x_val, x_test, input_shape = select_arrays(model_name, data)

    builder = get_model_builder(model_name)
    model = builder(input_shape, data.nb_classes)
    compile_model(model, args.learning_rate)

    run_name = f"{model_name}_seed{seed}"
    checkpoint_path = output_dirs["models"] / f"{run_name}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    if not args.no_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=100,
                restore_best_weights=True,
            )
        )

    start_train = time.perf_counter()
    history = model.fit(
        x_train,
        data.y_train,
        validation_data=(x_val, data.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - start_train

    # Load the checkpoint selected on validation loss.
    model = tf.keras.models.load_model(checkpoint_path)

    start_inference = time.perf_counter()
    y_pred_probabilities = model.predict(x_test, verbose=0)
    inference_time = time.perf_counter() - start_inference

    metrics = evaluate_classifier(data.y_test_labels, y_pred_probabilities)

    save_confusion_matrix(
        metrics.confusion_matrix,
        output_dirs["confusion_matrices"] / f"{run_name}.csv",
    )
    plot_training_curves(
        history,
        output_dirs["training_curves"] / run_name,
    )

    return {
        "model": model_name,
        "seed": seed,
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision,
        "recall_macro": metrics.recall,
        "f1_macro": metrics.f1,
        "params": model.count_params(),
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "inference_time_per_sample_sec": inference_time / len(x_test),
        "model_size_mb": model_file_size_mb(checkpoint_path),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "batch_size": args.batch_size,
        "validation_size": args.validation_size,
        "early_stopping": not args.no_early_stopping,
    }


def append_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    """Append one row to metrics.csv, creating the header if needed."""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_path.exists()

    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.results_dir)

    if args.overwrite_results:
        clear_previous_results(output_dirs)

    metrics_path = output_dirs["root"] / "metrics.csv"

    for seed in args.seeds:
        for model_name in args.models:
            row = train_one_model(model_name, seed, args, output_dirs)
            append_metrics(metrics_path, row)
            print(
                f"{model_name} seed={seed} "
                f"accuracy={row['accuracy']:.4f} "
                f"f1={row['f1_macro']:.4f} "
                f"params={row['params']}"
            )


if __name__ == "__main__":
    main()
