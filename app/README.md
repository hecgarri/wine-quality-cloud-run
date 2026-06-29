# Aplicación Streamlit y Docker

Esta carpeta contiene la interfaz Streamlit, el entorno Poetry de inferencia y
el Dockerfile. La aplicación carga `model.keras`, recopila 12 entradas y estima
la calidad del vino en una escala de 0 a 10.

## Funcionalidad

La interfaz incluye:

- un selector para vinos tintos o blancos;
- 11 controles para propiedades fisicoquímicas;
- rangos calculados desde Wine Quality de UCI;
- valores de respaldo cuando el dataset no está disponible;
- una predicción numérica y una interpretación descriptiva.

`@st.cache_resource` evita recargar el modelo en cada interacción.
`@st.cache_data` evita descargar y procesar repetidamente el dataset.

## Archivo requerido

La construcción necesita `app/model.keras`. Git ignora ese archivo porque es un
artefacto entrenado.

Comprueba su existencia desde `app/`:

```powershell
Get-Item .\model.keras
```

Si falta, vuelve a la raíz y genera el modelo con Docker:

```powershell
cd ..
docker run --rm `
  -v "${PWD}:/workspace" -w /workspace/parte0 `
  python:3.12-slim `
  sh -c "pip install --quiet poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root && poetry run python train.py"
```

También puedes generarlo con Poetry desde `parte0/`:

```powershell
cd parte0
poetry install
poetry run python train.py
```

## Ejecutar localmente con Poetry

Ejecuta desde `app/`:

```powershell
poetry install
poetry run streamlit run app.py
```

Abre http://127.0.0.1:8501. Detén Streamlit con `Ctrl+C`.

## Construir la imagen Docker

Ejecuta desde `app/`:

```powershell
docker build -t wine-quality-app .
```

La construcción instala Poetry dentro de `python:3.12-slim`, instala las
dependencias bloqueadas y copia `app.py` y `model.keras`.

## Ejecutar el contenedor

```powershell
docker run --rm --name wine-quality-app -p 8501:8501 wine-quality-app
```

Abre http://127.0.0.1:8501. Streamlit escucha en `0.0.0.0:8501` dentro del
contenedor, como exige el enunciado.

Detén el contenedor con `Ctrl+C`. Si lo ejecutaste en segundo plano, utiliza:

```powershell
docker stop wine-quality-app
```

## Verificar el contenedor

Comprueba el estado:

```powershell
docker ps --filter name=wine-quality-app
```

Consulta los logs:

```powershell
docker logs wine-quality-app
```

Comprueba el endpoint de salud de Streamlit:

```powershell
(Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8501/_stcore/health").Content
```

La respuesta esperada es `ok`.

## Desplegar en Cloud Run

El script de la raíz publica el modelo en Cloud Storage, construye la imagen con
Cloud Build y la almacena en Artifact Registry.

Ejecuta desde la raíz:

```powershell
.\deploy-cloud-run.ps1 -DryRun
.\deploy-cloud-run.ps1
```

Consulta el [README principal](../README.md) para revisar parámetros, versionado
del modelo y configuración de escalado.

## Dependencias de inferencia

El entorno de `app/` incluye:

- Streamlit para la interfaz web.
- TensorFlow CPU para cargar y ejecutar Keras.
- Pandas y NumPy para preparar las entradas.

Optuna y MLflow pertenecen a `parte0/` y no se instalan en la imagen de la
aplicación.

## Solución de problemas

### Docker no encuentra `model.keras`

Ejecuta el entrenamiento y vuelve a construir la imagen. El Dockerfile no puede
completar `COPY model.keras .` si falta el archivo.

### El puerto 8501 está ocupado

Busca el proceso o contenedor que utiliza el puerto:

```powershell
docker ps
Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
```

Detén el contenedor anterior o publica temporalmente otro puerto del host:

```powershell
docker run --rm -p 8502:8501 wine-quality-app
```

En ese caso, abre http://127.0.0.1:8502.

### La descarga de UCI falla

La aplicación utiliza estadísticas locales de respaldo para configurar los
controles. La predicción continúa funcionando porque el modelo ya contiene la
normalización.

### El contenedor se detiene

Consulta `docker logs wine-quality-app`. Los errores más comunes son un modelo
ausente, un archivo incompatible o memoria insuficiente para TensorFlow.
