"""Model definitions for ECG200 classification."""

from __future__ import annotations

import tensorflow as tf


def build_mlp(input_shape: tuple[int, ...], nb_classes: int) -> tf.keras.Model:
    """Build a simple MLP baseline.

    The MLP receives the ECG signal flattened as a vector and learns dense
    nonlinear combinations of all time steps.
    """

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP")


def build_cnn(input_shape: tuple[int, ...], nb_classes: int) -> tf.keras.Model:
    """Build a compact 1D CNN.

    Conv1D is appropriate for ECG signals because it can learn local temporal
    motifs while sharing weights across the whole time axis.
    """

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN1D")


def build_rnn(input_shape: tuple[int, ...], nb_classes: int) -> tf.keras.Model:
    """Placeholder for the RNN/LSTM model.

    To be implemented by the teammate responsible for the RNN part.

    Suggested starting point:

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.LSTM(32)(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
    """

    raise NotImplementedError(
        "RNN/LSTM model not implemented yet. "
        "Implement build_rnn() in src/models.py before training with --models rnn."
    )


def get_model_builder(model_name: str):
    """Return the model builder associated with a model name."""

    builders = {
        "mlp": build_mlp,
        "cnn": build_cnn,
        "rnn": build_rnn,
    }

    try:
        return builders[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(builders))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc
