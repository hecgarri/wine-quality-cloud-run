"""Busca hiperparámetros con Optuna y registra cada trial en MLflow."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import optuna
from tensorflow import keras

from train import load_data, split_data, train_model

BASE_DIR = Path(__file__).resolve().parent
MLFLOW_DB = BASE_DIR / "mlflow.db"
OPTUNA_DB = BASE_DIR / "optuna.db"
ARTIFACTS_DIR = BASE_DIR / "mlruns"
EXPERIMENT_NAME = "wine-quality-optuna"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=45)
    return parser.parse_args()


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME, ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def main() -> None:
    args = parse_args()
    configure_mlflow()
    X_train, y_train, X_val, y_val, _, _ = split_data(load_data())

    def objective(trial: optuna.Trial) -> float:
        keras.backend.clear_session()
        params: dict[str, int | float] = {
            "units_1": trial.suggest_categorical("units_1", [32, 64, 96, 128]),
            "units_2": trial.suggest_categorical("units_2", [16, 32, 48, 64]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 3e-3, log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
            "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}"):
            mlflow.log_params(params)
            model, history = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                params,
                epochs=args.epochs,
                verbose=0,
            )
            val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
            mlflow.log_metrics(
                {
                    "val_loss": float(val_loss),
                    "val_mae": float(val_mae),
                    "epochs_ran": len(history.history["loss"]),
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                model_path = Path(tmp) / "model.keras"
                model.save(model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
            return float(val_mae)

    study = optuna.create_study(
        study_name="wine-quality",
        storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=0),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)
    print(f"Mejor MAE de validación: {study.best_value:.4f}")
    print("Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value!r}")


if __name__ == "__main__":
    main()
