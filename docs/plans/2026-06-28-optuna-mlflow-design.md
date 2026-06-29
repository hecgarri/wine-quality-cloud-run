# Diseño: experimentación con Optuna y MLflow

## Objetivo

Usar Optuna para seleccionar hiperparámetros y MLflow para registrar los
experimentos, sin convertirlos en dependencias de ejecución de Streamlit ni de
Docker.

## Flujo

`optimize.py` ejecutará trials reproducibles y registrará parámetros y métricas
en MLflow. Sus artefactos locales no se incluirán en la entrega. Los mejores
hiperparámetros se fijarán en `train.py`, que seguirá siendo el comando oficial
para entrenar y exportar `app/model.keras`.

La aplicación continuará deserializando el archivo Keras local. No consultará
MLflow durante la inferencia.

## Entrega

El README explicará que la optimización es adicional y que la bondad de ajuste
no forma parte del criterio principal de evaluación. El ZIP final contendrá
únicamente las carpetas `parte0/` y `app/`, con los tipos de archivo permitidos
por el enunciado.

## Verificación

Se ejecutarán la optimización, el entrenamiento definitivo, la carga del modelo,
la construcción de Docker y una auditoría del contenido del ZIP.
