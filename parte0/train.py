"""Entrena y serializa un modelo Keras para predecir la calidad de un vino."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 0
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]
TARGET = "quality"

BEST_PARAMS: dict[str, int | float] = {
    "units_1": 128,
    "units_2": 64,
    "learning_rate": 0.0008019358005221843,
    "dropout": 0.2,
    "l2": 0.0006780227021579834,
    "batch_size": 32,
}


def set_reproducible_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)


def load_data() -> pd.DataFrame:
    red = pd.read_csv(RED_URL, sep=";")
    white = pd.read_csv(WHITE_URL, sep=";")
    red["wine_type"] = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)


def split_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[FEATURES].astype("float32").values
    y = df[TARGET].astype("float32").values
    perm = np.random.default_rng(SEED).permutation(len(X))
    n_test = int(0.2 * len(X))
    n_val = int(0.2 * (len(X) - n_test))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def build_model(train_features: np.ndarray, params: dict[str, int | float]) -> keras.Model:
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_features)
    regularizer = keras.regularizers.l2(float(params["l2"]))
    layers: list[keras.layers.Layer] = [
        normalizer,
        keras.layers.Dense(
            int(params["units_1"]), activation="relu", kernel_regularizer=regularizer
        ),
    ]
    if float(params["dropout"]) > 0:
        layers.append(keras.layers.Dropout(float(params["dropout"])))
    layers.append(
        keras.layers.Dense(
            int(params["units_2"]), activation="relu", kernel_regularizer=regularizer
        )
    )
    if float(params["dropout"]) > 0:
        layers.append(keras.layers.Dropout(float(params["dropout"])))
    layers.append(keras.layers.Dense(1))

    model = keras.Sequential(
        [keras.Input(shape=(train_features.shape[1],), name="features"), *layers],
        name="wine_quality_regressor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, int | float],
    *,
    epochs: int = 60,
    verbose: int = 2,
) -> tuple[keras.Model, keras.callbacks.History]:
    set_reproducible_seed()
    model = build_model(X_train, params)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=int(params["batch_size"]),
        verbose=verbose,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=8, restore_best_weights=True, monitor="val_loss"
            )
        ],
    )
    return model, history


def main() -> None:
    set_reproducible_seed()
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    print(f"Dataset Wine Quality: {len(df)} filas")
    print(
        f"Entrenamiento: {len(X_train)} | Validación: {len(X_val)} | "
        f"Prueba: {len(X_test)}"
    )

    model, _ = train_model(X_train, y_train, X_val, y_val, BEST_PARAMS)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"MSE en prueba: {test_loss:.3f}")
    print(f"MAE en prueba: {test_mae:.3f} puntos de calidad")

    out = Path(__file__).resolve().parent.parent / "app" / "model.keras"
    model.save(out)
    reloaded = keras.models.load_model(out)
    preview = reloaded.predict(X_test[:3], verbose=0).ravel()
    print(f"Modelo serializado en: {out}")
    print(f"Verificación de recarga: {np.round(preview, 2)}")


if __name__ == "__main__":
    main()
