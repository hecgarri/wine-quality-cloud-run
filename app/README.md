# Parte 1 + 2 — Aplicación Streamlit dockerizada (Wine Quality)

Aplicación **Streamlit** que deserializa el modelo **Keras** entrenado en la
Parte 0 (`model.keras`) y predice la **calidad** de un vino (escala 0–10) a
partir de un formulario con su análisis fisicoquímico.

- `@st.cache_resource` → carga del modelo.
- `@st.cache_data` → carga del dataset (para fijar los rangos de los widgets).
- Formulario con **12 entradas** (tipo de vino + 11 propiedades) que mapean a las
  features del modelo, mostradas con widgets apropiados (`st.selectbox`,
  `st.slider`).
- La salida se presenta de forma **amigable**: valor `X.X / 10` + interpretación
  (Baja / Media / Buena / Excelente) y una explicación de qué significa.

> El archivo `model.keras` lo genera y copia aquí automáticamente el script
> `../parte0/train.py`. Si aún no existe, ejecuta primero la Parte 0.

## Requisitos

- Python 3.12 + [Poetry](https://python-poetry.org/) (para ejecución local).
- [Docker](https://docs.docker.com/) (para la Parte 2).

## Ejecución local con Poetry

```bash
poetry install
poetry run streamlit run app.py
```

Luego abre http://localhost:8501 en el navegador.

## Ejecución con Docker (Parte 2)

```bash
# Construir la imagen
docker build -t hector-garrido .

# Levantar el contenedor (Streamlit escucha en 0.0.0.0:8501)
docker run -p 8501:8501 -d hector-garrido
```

Luego abre http://localhost:8501 en el navegador.
