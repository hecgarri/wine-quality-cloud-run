# Evaluación Práctica 2

Predictor de calidad de vino basado en Keras, presentado mediante Streamlit y
empaquetado como una imagen Docker.

## Alcance

La entrega conserva dos entornos Poetry independientes:

- `parte0/`: entrenamiento, optimización y serialización del modelo.
- `app/`: inferencia mediante Streamlit y construcción de la imagen Docker.

Optuna y MLflow se utilizaron como herramientas complementarias vistas en
clases. La bondad de ajuste no es el foco de la evaluación; el flujo obligatorio
sigue siendo entrenar, serializar y ejecutar la aplicación sin errores.

## Entrenar el modelo definitivo

```powershell
cd parte0
poetry install
poetry run python train.py
```

El comando descarga Wine Quality desde UCI y genera `app/model.keras`. Este
archivo es necesario para Docker y para Cloud Run, pero se mantiene fuera de Git.

## Ejecutar la optimización

```powershell
cd parte0
poetry install
poetry run python optimize.py --trials 10
```

Optuna persiste el estudio `wine-quality` en `optuna.db`. MLflow registra cada
trial en el experimento `wine-quality-optuna`, incluyendo hiperparámetros,
`val_mae`, `val_loss`, número de épocas y el modelo resultante.

La búsqueda utilizada para el modelo entregado ejecutó 10 trials. El mejor
obtuvo un MAE de validación de `0.5371`; el modelo definitivo obtuvo un MAE de
prueba de `0.551`.

## Explorar los experimentos con MLflow

Después de ejecutar `optimize.py`, inicia el servidor desde `parte0/`:

```powershell
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre http://127.0.0.1:5000, selecciona `wine-quality-optuna` y:

1. Ordena los runs por `val_mae` ascendente.
2. Selecciona varios trials y usa **Compare**.
3. Compara `learning_rate`, `units_1`, `units_2`, `dropout`, `l2` y
   `batch_size`.
4. Revisa `val_mae`, `val_loss`, `epochs_ran` y el artefacto
   `model/model.keras`.

## Explorar el estudio de Optuna

Desde `parte0/`, abre la base SQLite con la imagen oficial de Optuna Dashboard:

```powershell
docker run --rm -p 127.0.0.1:8080:8080 `
  -v "${PWD}:/app" -w /app `
  ghcr.io/optuna/optuna-dashboard sqlite:///optuna.db
```

Abre http://127.0.0.1:8080 y selecciona `wine-quality`. El dashboard permite
revisar el historial de optimización, la importancia de los hiperparámetros, las
relaciones entre parámetros y la tabla completa de trials.

`mlflow.db`, `optuna.db` y `mlruns/` son artefactos locales de experimentación y
no se versionan en Git.

## Ejecutar Streamlit con Docker

```powershell
cd app
docker build -t hector-garrido .
docker run --rm -p 8501:8501 hector-garrido
```

La aplicación queda disponible en http://localhost:8501.

## Desplegar en Cloud Run

Requisitos:

- Google Cloud CLI autenticado.
- Un proyecto seleccionado mediante `gcloud config set project`.
- `app/model.keras` generado localmente.

Valida el despliegue sin crear recursos:

```powershell
.\deploy-cloud-run.ps1 -DryRun
```

Ejecuta el despliegue:

```powershell
.\deploy-cloud-run.ps1
```

El script utiliza el proyecto activo y, por defecto:

- publica el modelo en `gs://<project-id>-ml-models/wine-quality/<sha256>/`;
- construye la imagen con Cloud Build;
- almacena versiones en Artifact Registry;
- despliega `wine-quality-predictor` en `us-central1`;
- permite acceso público;
- configura facturación por solicitud, mínimo cero y máximo una instancia.

Los valores se pueden reemplazar mediante los parámetros `ProjectId`, `Region`,
`ServiceName`, `ArtifactRepository`, `ImageName`, `ModelBucket` y `ModelPath`.

## Estructura principal

```text
.
|-- parte0/
|   |-- train.py
|   |-- optimize.py
|   |-- pyproject.toml
|   |-- poetry.lock
|   `-- README.md
|-- app/
|   |-- app.py
|   |-- Dockerfile
|   |-- pyproject.toml
|   |-- poetry.lock
|   `-- README.md
|-- deploy-cloud-run.ps1
`-- README.md
```
