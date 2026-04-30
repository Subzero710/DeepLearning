"""Dataset loading and preprocessing for ECG200."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


TRAIN_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv"
TEST_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv"


@dataclass(frozen=True)
class ECGData:
    """Container for all prepared ECG200 arrays."""

    x_train_mlp: np.ndarray
    x_val_mlp: np.ndarray
    x_test_mlp: np.ndarray
    x_train_seq: np.ndarray
    x_val_seq: np.ndarray
    x_test_seq: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_labels: np.ndarray
    y_val_labels: np.ndarray
    y_test_labels: np.ndarray
    class_names: list[str]
    input_shape_mlp: tuple[int, ...]
    input_shape_seq: tuple[int, ...]
    nb_classes: int


def download_ecg200(data_dir: str | Path = "data") -> tuple[Path, Path]:
    """Download ECG200 train/test TSV files if they are missing."""

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_path = data_path / "ECG200_TRAIN.tsv"
    test_path = data_path / "ECG200_TEST.tsv"

    if not train_path.exists():
        urllib.request.urlretrieve(TRAIN_URL, train_path)

    if not test_path.exists():
        urllib.request.urlretrieve(TEST_URL, test_path)

    return train_path, test_path


def _read_ecg_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a UCR-style TSV file.

    ECG200 files do not contain a header. The first column is the class label
    and the remaining columns are the time-series values.
    """

    df = pd.read_csv(path, sep="\t", header=None).dropna()
    labels = df.iloc[:, 0].to_numpy()
    features = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return features, labels


def load_ecg200(
    data_dir: str | Path = "data",
    validation_size: float = 0.2,
    seed: int = 42,
) -> ECGData:
    """Download, split, encode and normalize ECG200.

    The scaler is fitted only on the training subset, not on validation or test.
    This prevents data leakage.
    """

    train_path, test_path = download_ecg200(data_dir)

    x_train_full, y_train_full_raw = _read_ecg_tsv(train_path)
    x_test_raw, y_test_raw = _read_ecg_tsv(test_path)

    label_encoder = LabelEncoder()
    y_train_full_labels = label_encoder.fit_transform(y_train_full_raw)
    y_test_labels = label_encoder.transform(y_test_raw)

    x_train_raw, x_val_raw, y_train_labels, y_val_labels = train_test_split(
        x_train_full,
        y_train_full_labels,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train_full_labels,
    )

    scaler = StandardScaler()
    x_train_mlp = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val_mlp = scaler.transform(x_val_raw).astype(np.float32)
    x_test_mlp = scaler.transform(x_test_raw).astype(np.float32)

    # Sequence format for Conv1D and RNN/LSTM: (samples, timesteps, channels).
    x_train_seq = x_train_mlp[..., np.newaxis]
    x_val_seq = x_val_mlp[..., np.newaxis]
    x_test_seq = x_test_mlp[..., np.newaxis]

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_train = one_hot_encoder.fit_transform(y_train_labels.reshape(-1, 1)).astype(np.float32)
    y_val = one_hot_encoder.transform(y_val_labels.reshape(-1, 1)).astype(np.float32)
    y_test = one_hot_encoder.transform(y_test_labels.reshape(-1, 1)).astype(np.float32)

    class_names = [str(value) for value in label_encoder.classes_]

    return ECGData(
        x_train_mlp=x_train_mlp,
        x_val_mlp=x_val_mlp,
        x_test_mlp=x_test_mlp,
        x_train_seq=x_train_seq,
        x_val_seq=x_val_seq,
        x_test_seq=x_test_seq,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_train_labels=y_train_labels,
        y_val_labels=y_val_labels,
        y_test_labels=y_test_labels,
        class_names=class_names,
        input_shape_mlp=x_train_mlp.shape[1:],
        input_shape_seq=x_train_seq.shape[1:],
        nb_classes=y_train.shape[1],
    )
