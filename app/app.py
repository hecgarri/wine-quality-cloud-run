"""Parte 1 - Aplicacion Streamlit: predictor de calidad de vino.

Deserializa el modelo Keras entrenado en la Parte 0 (model.keras) y lo usa para
predecir la calidad sensorial (escala 0-10) de un vino a partir de un formulario.

- @st.cache_resource -> carga del modelo (objeto pesado, se cachea una vez).
- @st.cache_data      -> carga del dataset (para fijar rangos de los widgets).

Uso local:
    poetry install
    poetry run streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(
    page_title="Predictor de Calidad de Vino",
    layout="centered",
)

# Fuentes publicas del dataset (mismas que en parte0/train.py)
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Orden de features que el modelo espera (debe coincidir con parte0/train.py)
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

# Metadatos amigables para cada widget: (etiqueta, unidad, paso, ayuda)
META = {
    "fixed acidity": ("Acidez fija", "g/L", 0.1, "Acidez no volátil (ácido tartárico)."),
    "volatile acidity": ("Acidez volátil", "g/L", 0.01, "Ácido acético; en exceso da sabor a vinagre."),
    "citric acid": ("Ácido cítrico", "g/L", 0.01, "Aporta frescura al vino."),
    "residual sugar": ("Azúcar residual", "g/L", 0.1, "Azúcar que queda tras la fermentación."),
    "chlorides": ("Cloruros", "g/L", 0.001, "Cantidad de sal en el vino."),
    "free sulfur dioxide": ("SO₂ libre", "mg/L", 1.0, "Previene oxidación y microbios."),
    "total sulfur dioxide": ("SO₂ total", "mg/L", 1.0, "SO₂ libre + combinado."),
    "density": ("Densidad", "g/cm³", 0.0005, "Depende del azúcar y el alcohol."),
    "pH": ("pH", "", 0.01, "Nivel de acidez (0 muy ácido - 14 muy básico)."),
    "sulphates": ("Sulfatos", "g/L", 0.01, "Aditivo antimicrobiano y antioxidante."),
    "alcohol": ("Alcohol", "% vol", 0.1, "Porcentaje de alcohol por volumen."),
}

# Features numericas (todas menos el tipo de vino)
NUMERIC_FEATURES = [f for f in FEATURES if f != "wine_type"]

# Rango (percentiles 1-99) y mediana por feature, calculados sobre el dataset.
# Sirven de respaldo para los sliders si no hubiera conexion para descargar.
FALLBACK_STATS: dict[str, tuple[float, float, float]] = {
    "fixed acidity": (5.1, 12.0, 7.0),
    "volatile acidity": (0.12, 0.88, 0.29),
    "citric acid": (0.0, 0.74, 0.31),
    "residual sugar": (0.9, 18.2, 3.0),
    "chlorides": (0.021, 0.1862, 0.047),
    "free sulfur dioxide": (4.0, 77.0, 29.0),
    "total sulfur dioxide": (11.0, 238.0, 118.0),
    "density": (0.9889, 1.0006, 0.9949),
    "pH": (2.89, 3.64, 3.21),
    "sulphates": (0.3, 0.99, 0.51),
    "alcohol": (8.7, 13.4, 10.3),
}


@st.cache_resource
def load_model() -> keras.Model:
    """Carga (una sola vez) el modelo Keras serializado."""
    return keras.models.load_model("model.keras")


@st.cache_data
def load_feature_stats() -> dict[str, tuple[float, float, float]]:
    """Descarga el dataset publico (cacheado) y calcula, por cada feature
    numerica, su rango (percentiles 1 y 99) y su mediana para configurar los
    sliders. Si la descarga falla (p. ej. sin conexion), usa valores predefinidos
    para que la aplicacion siga funcionando."""
    try:
        red = pd.read_csv(RED_URL, sep=";")
        white = pd.read_csv(WHITE_URL, sep=";")
        red["wine_type"] = 0
        white["wine_type"] = 1
        df = pd.concat([red, white], ignore_index=True)
        return {
            f: (
                round(float(df[f].quantile(0.01)), 4),
                round(float(df[f].quantile(0.99)), 4),
                round(float(df[f].median()), 4),
            )
            for f in NUMERIC_FEATURES
        }
    except Exception:
        return FALLBACK_STATS


def interpret(score: float) -> tuple[str, str, str]:
    """Devuelve (titulo, mensaje, color) segun el puntaje estimado."""
    if score < 5:
        return "Calidad baja", "Por debajo del promedio de los vinos del set.", "off"
    if score < 6:
        return "Calidad media", "Un vino correcto, sin destacar.", "normal"
    if score < 7:
        return "Buena calidad", "Mejor que la mayoría de los vinos del set.", "normal"
    return "Excelente calidad", "Entre los vinos mejor evaluados del set.", "normal"


def main() -> None:
    st.title("Predictor de Calidad de Vino")
    st.markdown(
        "Esta aplicación estima la **calidad sensorial** de un vino (escala "
        "**0 a 10**) a partir de su análisis fisicoquímico, usando una **red "
        "neuronal (Keras)** entrenada sobre el dataset *Wine Quality* (UCI), que "
        "reúne vinos tintos y blancos portugueses *Vinho Verde*.\n\n"
        "Ajusta los valores del análisis de laboratorio en el formulario y pulsa "
        "**Predecir calidad**."
    )
    st.markdown("---")

    stats = load_feature_stats()
    model = load_model()

    st.subheader("Análisis del vino")

    tipo = st.selectbox(
        "Tipo de vino",
        options=["Tinto", "Blanco"],
        index=0,
        help="Color del vino analizado.",
    )
    wine_type_val = 0 if tipo == "Tinto" else 1

    values: dict[str, float] = {}
    cols = st.columns(2)
    for i, feat in enumerate(NUMERIC_FEATURES):
        label, unit, step, helptext = META[feat]
        lo, hi, med = stats[feat]
        widget_label = f"{label} ({unit})" if unit else label
        with cols[i % 2]:
            values[feat] = st.slider(
                widget_label,
                min_value=lo,
                max_value=hi,
                value=med,
                step=step,
                help=helptext,
            )

    st.markdown("---")

    if st.button("Predecir calidad", type="primary", use_container_width=True):
        values["wine_type"] = float(wine_type_val)
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES).astype("float32")
        raw_score = float(model.predict(row.values, verbose=0)[0][0])
        score = max(0.0, min(10.0, raw_score))  # acotar a la escala 0-10

        titulo, mensaje, _ = interpret(score)

        st.subheader("Resultado")
        st.metric(label="Calidad estimada", value=f"{score:.1f} / 10")
        st.progress(score / 10.0)
        st.success(f"**{titulo}** — {mensaje}")
        st.caption(
            "La *calidad* es el puntaje sensorial promedio que asignan catadores "
            "expertos (0 = muy malo, 10 = excelente). El valor mostrado es la "
            "estimación del modelo para un vino con las características ingresadas."
        )


if __name__ == "__main__":
    main()
