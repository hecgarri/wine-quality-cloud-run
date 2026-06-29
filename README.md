# Predictor de calidad de vino

Este proyecto entrena una red neuronal Keras para estimar la calidad de vinos
tintos y blancos. Una aplicación Streamlit carga el modelo, recibe propiedades
fisicoquímicas y presenta la predicción en una escala de 0 a 10.

La solución incluye entrenamiento reproducible, búsqueda de hiperparámetros con
Optuna, seguimiento de experimentos con MLflow, ejecución con Docker y un script
de despliegue para Google Cloud Run.

## Qué enseña este proyecto

El repositorio muestra el recorrido completo de un modelo de machine learning,
no solamente su entrenamiento. Al terminar el recorrido puedes identificar qué
herramienta responde cada pregunta:

- ¿Cómo se preparan los datos y se entrena una red neuronal?
- ¿Cómo se repite el entrenamiento con las mismas dependencias?
- ¿Cómo se comparan configuraciones sin perder resultados anteriores?
- ¿Cómo se convierte un modelo en una aplicación web?
- ¿Cómo se empaqueta la aplicación para ejecutarla en otro computador?
- ¿Cómo se versionan por separado el código, el modelo y la imagen desplegable?
- ¿Cómo se publica la aplicación sin mantener un servidor encendido?

## Modelo mental: cinco tipos de objetos

Un proyecto de machine learning produce varios objetos con ciclos de vida
distintos. Confundirlos suele llevar a repositorios pesados o despliegues poco
reproducibles.

| Objeto | Ejemplo en este proyecto | Sistema responsable |
| --- | --- | --- |
| Código fuente | `train.py`, `app.py` | Git y GitHub |
| Definición del entorno | `pyproject.toml`, `poetry.lock` | Poetry |
| Historial experimental | trials, parámetros y métricas | Optuna y MLflow |
| Modelo entrenado | `model.keras` | Cloud Storage |
| Aplicación empaquetada | imagen Docker | Artifact Registry |
| Aplicación en ejecución | servicio web público | Cloud Run |

Git conserva archivos de texto y su historial. Cloud Storage conserva el modelo
binario. Artifact Registry conserva las imágenes Docker. Cloud Run ejecuta una
versión concreta de esas imágenes.

## Flujo de extremo a extremo

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

El flujo sigue estos pasos:

1. Pandas descarga y combina los CSV de UCI.
2. NumPy transforma las columnas en matrices numéricas.
3. Keras define la red y TensorFlow ejecuta el entrenamiento.
4. Optuna prueba combinaciones de hiperparámetros.
5. MLflow conserva los parámetros, métricas y modelos de cada trial.
6. `train.py` fija la configuración seleccionada y crea `model.keras`.
7. Streamlit carga el modelo y construye la interfaz de predicción.
8. Docker empaqueta código, dependencias y modelo en una imagen.
9. Cloud Storage conserva el modelo mediante una ruta identificada por SHA-256.
10. Cloud Build construye la imagen en Google Cloud.
11. Artifact Registry conserva cada versión de la imagen.
12. Cloud Run crea una revisión y atiende solicitudes HTTP.

`parte0/` y `app/` mantienen entornos Poetry independientes. El modelo y las
bases de experimentación se generan localmente y no se almacenan en Git.

## Herramientas y responsabilidades

### Pandas y NumPy: preparar datos tabulares

[Pandas](https://pandas.pydata.org/docs/) representa los CSV como tablas con
columnas nombradas. `train.py` lo usa para descargar los vinos tintos y blancos,
añadir el tipo de vino y seleccionar las variables de entrada.

[NumPy](https://numpy.org/doc/) representa los datos como matrices numéricas que
TensorFlow puede procesar eficientemente. También genera la permutación
reproducible que divide los datos en entrenamiento, validación y prueba.

Ambas herramientas aparecen en `parte0/train.py`. La aplicación también usa
Pandas para construir una fila con el mismo orden de variables que espera el
modelo.

### TensorFlow y Keras: definir y entrenar la red

[TensorFlow](https://www.tensorflow.org/) es el motor de cálculo numérico. Se
encarga de tensores, derivación automática, optimización y ejecución de las
operaciones que actualizan los pesos de la red.

[Keras](https://keras.io/) es la API de alto nivel utilizada para describir la
arquitectura. El proyecto crea una secuencia de capas de normalización, capas
densas, dropout y una salida lineal. Keras también proporciona `fit`,
`EarlyStopping`, `evaluate`, `save` y `load_model`.

La separación es conceptual: Keras expresa qué red quieres entrenar y
TensorFlow realiza el cálculo necesario. El resultado es `app/model.keras`, un
archivo que contiene arquitectura, pesos y normalización.

### Poetry: reproducir el entorno de Python

[Poetry](https://python-poetry.org/docs/) administra dependencias y entornos de
Python. `pyproject.toml` declara rangos y condiciones de plataforma;
`poetry.lock` fija las versiones exactas resueltas.

El proyecto mantiene dos entornos porque entrenamiento e inferencia tienen
responsabilidades diferentes:

- `parte0/` instala TensorFlow, Pandas, NumPy, Optuna y MLflow.
- `app/` instala TensorFlow CPU, Pandas, NumPy y Streamlit.

Esta separación evita que Docker instale herramientas experimentales que la
aplicación no necesita. `poetry install` reconstruye el entorno y `poetry run`
ejecuta un comando dentro de ese contexto.

### Optuna: buscar hiperparámetros

[Optuna](https://optuna.readthedocs.io/) automatiza la búsqueda de decisiones
que el entrenamiento no aprende por sí solo. En este proyecto explora tamaños de
capas, learning rate, dropout, regularización L2 y batch size.

Un **trial** es una ejecución con una combinación concreta. El **objective**
entrena el modelo y devuelve el MAE de validación. El **study** agrupa todos los
trials y conoce cuál obtuvo el menor error.

`parte0/optimize.py` conserva el estudio en `optuna.db`. Optuna responde qué
configuración conviene probar o seleccionar, pero no sustituye el entrenamiento
final ni el conjunto de prueba.

### MLflow: conservar evidencia experimental

[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) registra lo que
ocurre dentro de cada trial. Un **experiment** agrupa ejecuciones relacionadas y
un **run** contiene parámetros, métricas, etiquetas y artefactos.

El proyecto registra en `wine-quality-optuna`:

- los hiperparámetros propuestos por Optuna;
- `val_mae` y `val_loss`;
- el número de épocas ejecutadas;
- el archivo `model.keras` producido por el trial.

Optuna decide qué configuración evaluar; MLflow permite auditar y comparar lo
que ocurrió. `mlflow.db` guarda metadatos y `mlruns/` guarda artefactos. Ninguno
de esos archivos se publica en Git.

### Streamlit: convertir Python en una interfaz web

[Streamlit](https://docs.streamlit.io/) permite construir una aplicación web
con código Python. `app/app.py` crea selectores, sliders, textos y el botón que
invoca `model.predict`.

Streamlit vuelve a ejecutar el script cuando cambia un control. Por eso el
proyecto usa dos cachés:

- `st.cache_resource` mantiene una sola instancia del modelo cargado.
- `st.cache_data` conserva las estadísticas calculadas desde el dataset.

La aplicación no entrena. Solamente carga un modelo existente, valida el orden
de las 12 entradas y presenta una predicción comprensible.

### Docker: empaquetar una ejecución reproducible

[Docker](https://docs.docker.com/get-started/docker-concepts/) construye una
imagen a partir de `app/Dockerfile`. La imagen incluye Python 3.12, Poetry,
dependencias, `app.py` y una versión concreta de `model.keras`.

Una **imagen** es una plantilla inmutable. Un **contenedor** es un proceso creado
desde esa imagen. `docker build` crea la imagen y `docker run` inicia el
contenedor.

El puerto `8501` conecta el navegador del host con Streamlit dentro del
contenedor. La dirección interna `0.0.0.0` permite que Docker y Cloud Run envíen
tráfico al proceso.

### Git y GitHub: versionar el código

[Git](https://git-scm.com/doc) registra cambios locales mediante commits.
[GitHub](https://docs.github.com/) almacena el repositorio remoto y permite
compartir su historial.

El modelo no se guarda en Git porque es un artefacto binario derivado del
entrenamiento. El repositorio conserva el código y las versiones de dependencias
necesarias para regenerarlo. `.gitignore` evita publicar modelos, bases SQLite,
cachés y credenciales.

### Cloud Storage: versionar el modelo

[Cloud Storage](https://cloud.google.com/storage/docs) almacena objetos binarios.
`deploy-cloud-run.ps1` calcula el SHA-256 del modelo y utiliza una ruta como:

```text
gs://<bucket>/wine-quality/<sha256>/model.keras
```

El hash identifica exactamente el contenido. Dos modelos distintos producen
rutas distintas y una ejecución repetida con el mismo modelo reutiliza la misma
ruta. Esta estrategia evita depender de un nombre mutable como `latest`.

### Cloud Build: construir dentro de Google Cloud

[Cloud Build](https://cloud.google.com/build/docs/overview) ejecuta la
construcción Docker en infraestructura administrada. El script prepara un
contexto temporal, descarga desde GCS la versión exacta del modelo y envía ese
contexto al servicio.

La construcción remota evita depender de la imagen que exista en el computador
del desarrollador. Su salida es una imagen Docker etiquetada con fecha y hash del
modelo.

### Artifact Registry: conservar imágenes Docker

[Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview)
almacena la salida de Cloud Build. Cada imagen tiene un tag legible y un digest
SHA-256 inmutable.

Cloud Storage versiona el modelo como artefacto de machine learning. Artifact
Registry versiona la aplicación completa, que incluye código, entorno y una
copia de ese modelo.

### Cloud Run: ejecutar la aplicación bajo demanda

[Cloud Run](https://cloud.google.com/run/docs/overview/what-is-cloud-run)
ejecuta contenedores que atienden solicitudes HTTP. Cada despliegue crea una
revisión inmutable asociada a una imagen de Artifact Registry.

El script configura acceso público, facturación por solicitud, mínimo cero y
máximo una instancia. Con mínimo cero, Cloud Run puede apagar la instancia cuando
no recibe tráfico; la siguiente solicitud puede experimentar un arranque en frío.

Cloud Run no entrena ni modifica el modelo. Solamente inicia la imagen y dirige
solicitudes a Streamlit en el puerto `8501`.

### Google Cloud CLI: coordinar los servicios

[Google Cloud CLI](https://cloud.google.com/sdk/gcloud) proporciona el comando
`gcloud`. `deploy-cloud-run.ps1` lo usa para habilitar APIs, crear repositorios,
copiar el modelo, ejecutar Cloud Build y desplegar Cloud Run.

El parámetro `-DryRun` valida rutas, proyecto, hash y nombres sin crear recursos.
El despliegue real utiliza la cuenta autenticada y el proyecto activo.

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
