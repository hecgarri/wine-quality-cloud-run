# Parte 0 — Entrenamiento y serialización del modelo (Keras + Poetry)

Entrena una red neuronal en **Keras** que estima la **calidad** de un vino
(escala 0–10) a partir de sus propiedades fisicoquímicas, y la serializa con el
**formato nativo** `model.keras`.

- **Dataset:** Wine Quality (UCI) — vinos tintos y blancos *Vinho Verde*.
- **Descarga:** 100 % programática desde un enlace público: el script descarga los CSV directamente desde la URL de UCI.
- **Reproducibilidad:** semilla `0` fijada en `random`, `numpy` y `tensorflow`.

## Alcance de la optimización

Optuna y MLflow se utilizaron durante el desarrollo para comparar
hiperparámetros y registrar los experimentos, tal como se trabajó en clases. Esta
optimización es complementaria: la bondad de ajuste no es el foco de la
evaluación y el comando principal sigue siendo un entrenamiento determinista que
genera el modelo Keras requerido.

La búsqueda ejecutada consideró 10 trials. La configuración seleccionada obtuvo
un MAE de validación de `0.5371` puntos antes del entrenamiento definitivo.

## Requisitos

- Python 3.12
- [Poetry](https://python-poetry.org/) instalado.

## Comandos

```bash
# 1. Instalar las dependencias de esta parte (entorno propio como buena práctica)
poetry install

# 2. Entrenar y serializar el modelo
poetry run python train.py
```

## Experimentación opcional

```bash
poetry run python optimize.py --trials 10
```

Optuna conserva el estudio en `optuna.db`; MLflow almacena el seguimiento en
`mlflow.db` y `mlruns/`. Estos artefactos de desarrollo no forman parte del ZIP
de entrega.

Para explorar los runs de MLflow:

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La interfaz queda disponible en http://127.0.0.1:5000. Selecciona el experimento
`wine-quality-optuna`, ordena por `val_mae` y utiliza **Compare** para contrastar
parámetros, métricas y modelos.

Para explorar el estudio de Optuna desde PowerShell:

```powershell
docker run --rm -p 127.0.0.1:8080:8080 `
  -v "${PWD}:/app" -w /app `
  ghcr.io/optuna/optuna-dashboard sqlite:///optuna.db
```

Abre http://127.0.0.1:8080 y selecciona el estudio `wine-quality`.

Al ejecutar `train.py` se:

1. Descargan y combinan los datos (tinto + blanco) desde la URL pública.
2. Entrena la red con los hiperparámetros seleccionados y una capa
   `keras.layers.Normalization` adaptada **dentro** del modelo.
3. Serializa el modelo con el formato nativo (`model.save(...)`) **directamente en
   `../app/model.keras`**, de modo que viaje con el contenedor de la aplicación.
   Así `parte0/` se mantiene limpio (solo código, docs y archivos de Poetry).
4. Verifica el *round-trip* recargándolo (`keras.models.load_model`) y prediciendo.

> El modelo es autocontenido: la normalización está incorporada al propio
> modelo, por lo que la aplicación entrega los valores originales a `predict`.
