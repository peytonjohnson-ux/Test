"""
3-Layer CNN for EEG Seizure Detection from Scalogram Images.

Architecture
------------
Input  → Conv2D → BN → ReLU → MaxPool
       → Conv2D → BN → ReLU → MaxPool
       → Conv2D → BN → ReLU → GlobalAvgPool
       → Dense (dropout) → Sigmoid output

Supports configurable filters, kernel sizes, dense units, dropout and L2.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Sequence


def build_seizure_cnn(
    input_shape: tuple[int, int, int],
    filters: Sequence[int] = (32, 64, 128),
    kernel_sizes: Sequence[int] = (3, 3, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.5,
    l2_lambda: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and compile the 3-layer seizure-detection CNN.

    Parameters
    ----------
    input_shape   : (height, width, channels), e.g. (32, 32, 1) or (64, 64, 1)
    filters       : number of filters for each of the 3 conv layers
    kernel_sizes  : kernel size for each conv layer
    dense_units   : units in the fully-connected head
    dropout_rate  : dropout probability before the dense layer
    l2_lambda     : L2 regularisation coefficient on conv and dense weights
    learning_rate : Adam initial learning rate

    Returns
    -------
    Compiled keras.Model
    """
    assert len(filters) == 3 and len(kernel_sizes) == 3, \
        "filters and kernel_sizes must each have exactly 3 elements."

    reg = regularizers.l2(l2_lambda)
    inp = keras.Input(shape=input_shape, name="scalogram_input")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters[0], kernel_sizes[0],
        padding="same", kernel_regularizer=reg, name="conv1"
    )(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters[1], kernel_sizes[1],
        padding="same", kernel_regularizer=reg, name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(
        filters[2], kernel_sizes[2],
        padding="same", kernel_regularizer=reg, name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=reg, name="dense")(x)
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=output, name="SeizureCNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(log_dir: str = "logs", patience: int = 10) -> list:
    """
    Standard training callbacks:
    - EarlyStopping on val_auc
    - ModelCheckpoint  saves the best weights
    - ReduceLROnPlateau halves LR when val_loss stalls
    - TensorBoard      for live monitoring
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{log_dir}/best_model.keras",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
    ]
