# Predictor de calidad de vino

Este proyecto entrena una red neuronal Keras para estimar la calidad de vinos
tintos y blancos. Una aplicación Streamlit carga el modelo, recibe propiedades
fisicoquímicas y presenta la predicción en una escala de 0 a 10.

La solución incluye entrenamiento reproducible, búsqueda de hiperparámetros con
Optuna, seguimiento de experimentos con MLflow, ejecución con Docker y un script
de despliegue para Google Cloud Run.

## Arquitectura

```text
Wine Quality (UCI)
        |
        v
parte0/train.py ------> app/model.keras
        |                       |
        |                       v
        |                 app/app.py
        |                       |
        v                       v
Optuna + MLflow          Docker / Cloud Run
```

`parte0/` y `app/` mantienen entornos Poetry independientes. El modelo y las
bases de experimentación se generan localmente y no se almacenan en Git.

## Requisitos

Necesitas las siguientes herramientas según el flujo que quieras ejecutar:

- Docker Desktop para la aplicación, MLflow y Optuna Dashboard.
- Python 3.12 y Poetry para ejecutar el código directamente en Windows.
- Google Cloud CLI para desplegar en Cloud Run.

Comprueba las herramientas disponibles desde PowerShell:

```powershell
docker --version
python --version
poetry --version
gcloud.cmd --version
```

Si PowerShell no reconoce `poetry`, puedes instalarlo siguiendo la
[documentación oficial de Poetry](https://python-poetry.org/docs/#installation)
o utilizar las alternativas Docker de este README.

## Inicio rápido con Docker

El repositorio no incluye `app/model.keras`. Genera primero el modelo desde la
raíz del repositorio:

```powershell
docker run --rm `
  -v "${PWD}:/workspace" -w /workspace/parte0 `
  python:3.12-slim `
  sh -c "pip install --quiet poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && poetry run python train.py"
```

Después construye y ejecuta la aplicación:

```powershell
cd app
docker build -t wine-quality-app .
docker run --rm --name wine-quality-app -p 8501:8501 wine-quality-app
```

Abre http://127.0.0.1:8501. Detén la aplicación con `Ctrl+C`.

## Entrenamiento con Poetry

Ejecuta estos comandos desde la raíz:

```powershell
cd parte0
poetry install
poetry run python train.py
```

El script descarga el dataset, entrena el modelo, evalúa el conjunto de prueba y
escribe `app/model.keras`. Consulta [parte0/README.md](parte0/README.md) para
conocer la arquitectura, la partición de datos y los resultados obtenidos.

## Optimización con Optuna y MLflow

La optimización es complementaria al enunciado. Optuna busca hiperparámetros y
MLflow registra cada trial con sus parámetros, métricas y modelo.

Con Poetry, ejecuta desde `parte0/`:

```powershell
poetry install
poetry run python optimize.py --trials 10
```

Sin Poetry, ejecuta desde `parte0/`:

```powershell
docker run --rm `
  -v "${PWD}:/experiment" -w /experiment `
  python:3.12-slim `
  sh -c "pip install --quiet poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && poetry run python optimize.py --trials 10"
```

El proceso genera:

- `optuna.db`: estudio `wine-quality`.
- `mlflow.db`: metadatos del experimento `wine-quality-optuna`.
- `mlruns/`: modelos producidos por los trials.

## Explorar MLflow sin Poetry

El siguiente comando funciona con los experimentos existentes en `parte0/`.
Ejecuta el comando desde esa carpeta:

```powershell
docker run --rm -p 127.0.0.1:5000:5000 `
  -v "${PWD}:/experiment" -w /experiment `
  ghcr.io/mlflow/mlflow:v3.14.0 `
  mlflow server `
  --backend-store-uri sqlite:////experiment/mlflow.db `
  --default-artifact-root file:///experiment/mlruns `
  --host 0.0.0.0 --port 5000
```

Abre http://127.0.0.1:5000 y selecciona `wine-quality-optuna`.

1. Ordena los runs por `val_mae` ascendente.
2. Selecciona varios trials y usa **Compare**.
3. Contrasta `learning_rate`, `units_1`, `units_2`, `dropout`, `l2` y
   `batch_size`.
4. Revisa `val_mae`, `val_loss`, `epochs_ran` y `model/model.keras`.

Detén MLflow con `Ctrl+C`. Docker elimina el contenedor automáticamente.

## Explorar Optuna

Ejecuta Optuna Dashboard desde `parte0/`:

```powershell
docker run --rm -p 127.0.0.1:8080:8080 `
  -v "${PWD}:/app" -w /app `
  ghcr.io/optuna/optuna-dashboard sqlite:///optuna.db
```

Abre http://127.0.0.1:8080 y selecciona `wine-quality`. Revisa el historial, la
importancia de hiperparámetros, las relaciones entre parámetros y la tabla de
trials. Detén el dashboard con `Ctrl+C`.

## Resultados registrados

La búsqueda ejecutó 10 trials. El mejor trial alcanzó un MAE de validación de
`0.5371`. El modelo definitivo obtuvo un MAE de prueba de `0.551`.

Estos valores documentan la ejecución realizada, pero la bondad de ajuste no es
el foco principal de la evaluación.

## Despliegue en Cloud Run

El script [deploy-cloud-run.ps1](deploy-cloud-run.ps1) mantiene el modelo fuera
de Git y lo publica en una ruta GCS identificada por su SHA-256. Cloud Build
incorpora esa versión en una imagen almacenada en Artifact Registry.

Valida primero el despliegue sin crear recursos:

```powershell
.\deploy-cloud-run.ps1 -DryRun
```

Después ejecuta el despliegue real:

```powershell
.\deploy-cloud-run.ps1
```

La configuración predeterminada despliega `wine-quality-predictor` en
`us-central1`, permite acceso público y configura mínimo cero y máximo una
instancia.

## Archivos generados e ignorados

Git ignora los siguientes artefactos:

```text
app/model.keras
parte0/mlflow.db
parte0/optuna.db
parte0/mlruns/
entrega.zip
__pycache__/
```

`entrega.zip` conserva el modelo porque el enunciado exige incluirlo, aunque el
repositorio Git no lo versiona.

## Solución de problemas

### PowerShell no reconoce Poetry

Usa los comandos Docker anteriores o instala Poetry y abre una terminal nueva.
Comprueba después la instalación con `poetry --version`.

### Falta `app/model.keras`

Ejecuta `parte0/train.py` con Poetry o utiliza el comando Docker de inicio
rápido. Docker no puede construir la aplicación sin ese archivo.

### El puerto ya está ocupado

Identifica y detén el contenedor anterior:

```powershell
docker ps
docker stop wine-quality-app
```

### MLflow no muestra experimentos

Comprueba que ejecutas el comando desde `parte0/` y que existe `mlflow.db`:

```powershell
Get-Item .\mlflow.db
```

Si el archivo no existe, ejecuta primero `optimize.py`.

## Estructura

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
