# Entrenamiento y experimentación

Esta carpeta contiene el entrenamiento reproducible del predictor de calidad de
vino y el flujo opcional de optimización con Optuna y MLflow. El script principal
genera el modelo Keras que consume la aplicación Streamlit.

## Dataset

`train.py` descarga automáticamente los archivos de vinos tintos y blancos del
dataset Wine Quality de UCI. Cada observación contiene 11 mediciones
fisicoquímicas, el tipo de vino y una calidad sensorial entre 0 y 10.

El modelo utiliza 12 entradas:

1. Acidez fija.
2. Acidez volátil.
3. Ácido cítrico.
4. Azúcar residual.
5. Cloruros.
6. Dióxido de azufre libre.
7. Dióxido de azufre total.
8. Densidad.
9. pH.
10. Sulfatos.
11. Alcohol.
12. Tipo de vino.

## Partición y reproducibilidad

El código fija la semilla `0` en Python, NumPy y TensorFlow. Una permutación
reproducible divide las 6.497 observaciones en:

- 4.159 filas de entrenamiento.
- 1.039 filas de validación.
- 1.299 filas de prueba.

El conjunto de prueba permanece fuera de la búsqueda de hiperparámetros.

## Arquitectura seleccionada

```text
12 entradas
    |
Normalización
    |
Dense(128, ReLU) + Dropout(0.2)
    |
Dense(64, ReLU) + Dropout(0.2)
    |
Dense(1, lineal)
```

La capa de normalización forma parte del archivo `model.keras`. La aplicación
puede entregar valores originales sin reproducir transformaciones externas.

Los hiperparámetros definitivos son:

| Parámetro | Valor |
| --- | ---: |
| Primera capa | 128 unidades |
| Segunda capa | 64 unidades |
| Learning rate | 0.0008019358 |
| Dropout | 0.2 |
| Regularización L2 | 0.0006780227 |
| Batch size | 32 |

## Requisitos

- Python 3.12.
- Poetry, para ejecución directa.
- Docker Desktop, como alternativa cuando Poetry no está instalado.
- Conexión a internet para descargar el dataset la primera vez.

## Entrenar con Poetry

Ejecuta desde `parte0/`:

```powershell
poetry install
poetry run python train.py
```

La ejecución muestra las métricas, guarda `../app/model.keras`, vuelve a cargar
el archivo y realiza tres predicciones de comprobación.

## Entrenar sin Poetry mediante Docker

Ejecuta desde la raíz del repositorio, no desde `parte0/`:

```powershell
docker run --rm `
  -v "${PWD}:/workspace" -w /workspace/parte0 `
  python:3.12-slim `
  sh -c "pip install --quiet poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && poetry run python train.py"
```

El volumen permite que el contenedor escriba el modelo directamente en
`app/model.keras`.

## Ejecutar la búsqueda con Optuna

Con Poetry, ejecuta desde `parte0/`:

```powershell
poetry run python optimize.py --trials 10
```

Puedes cambiar el número de trials y el máximo de épocas:

```powershell
poetry run python optimize.py --trials 20 --epochs 60
```

Sin Poetry, ejecuta desde `parte0/`:

```powershell
docker run --rm `
  -v "${PWD}:/experiment" -w /experiment `
  python:3.12-slim `
  sh -c "pip install --quiet poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && poetry run python optimize.py --trials 10"
```

Optuna reutiliza `optuna.db` si existe. Cada nueva ejecución añade trials al
estudio `wine-quality` en lugar de reemplazarlo.

## Explorar MLflow

La opción Docker funciona aunque Windows no tenga Poetry. Ejecuta desde
`parte0/`:

```powershell
docker run --rm -p 127.0.0.1:5000:5000 `
  -v "${PWD}:/experiment" -w /experiment `
  ghcr.io/mlflow/mlflow:v3.14.0 `
  mlflow server `
  --backend-store-uri sqlite:////experiment/mlflow.db `
  --default-artifact-root file:///experiment/mlruns `
  --host 0.0.0.0 --port 5000
```

Abre http://127.0.0.1:5000 y selecciona `wine-quality-optuna`. Ordena por
`val_mae`, compara trials y abre `model/model.keras` dentro de los artefactos.

Detén el servidor con `Ctrl+C`.

## Explorar Optuna Dashboard

Ejecuta desde `parte0/`:

```powershell
docker run --rm -p 127.0.0.1:8080:8080 `
  -v "${PWD}:/app" -w /app `
  ghcr.io/optuna/optuna-dashboard sqlite:///optuna.db
```

Abre http://127.0.0.1:8080 y selecciona `wine-quality`. El dashboard muestra el
historial, los parámetros importantes, gráficos de relación y todos los trials.

## Resultados

La búsqueda utilizada para la entrega ejecutó 10 trials. El mejor trial obtuvo
un MAE de validación de `0.5371`. El entrenamiento definitivo alcanzó un MAE de
prueba de `0.551`.

## Archivos generados

| Ruta | Contenido | Se versiona en Git |
| --- | --- | --- |
| `../app/model.keras` | Modelo definitivo | No |
| `mlflow.db` | Metadatos de MLflow | No |
| `mlruns/` | Modelos de los trials | No |
| `optuna.db` | Estudio de Optuna | No |

## Solución de problemas

### Poetry no está disponible

Utiliza los comandos Docker anteriores. Si instalas Poetry, cierra y vuelve a
abrir PowerShell antes de comprobar `poetry --version`.

### MLflow abre sin runs

Verifica que `mlflow.db` existe en el directorio actual. Ejecuta primero
`optimize.py` si todavía no has creado la base.

### El entrenamiento no descarga el dataset

Comprueba la conexión a internet y el acceso a
`archive.ics.uci.edu`. La aplicación dispone de estadísticas de respaldo, pero
el entrenamiento necesita descargar los CSV.

### TensorFlow tarda en importar

La primera importación y la instalación de dependencias pueden tardar varios
minutos. Docker reutiliza las capas descargadas en construcciones posteriores.
