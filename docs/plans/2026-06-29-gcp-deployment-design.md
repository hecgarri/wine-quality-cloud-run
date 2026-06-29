# Diseño: publicación en GitHub y despliegue en Cloud Run

## Alcance

El repositorio contendrá código, documentación y archivos de Poetry. Los modelos,
experimentos locales, bases SQLite, cachés y archivos de entrega quedarán fuera de
Git.

## Versionado y despliegue

`deploy-cloud-run.ps1` calculará el SHA-256 de `app/model.keras` y lo publicará en
una ruta inmutable de Cloud Storage. Después creará un contexto temporal con los
archivos de `app/`, recuperará desde GCS esa versión exacta del modelo y usará
Cloud Build para construir una imagen versionada en Artifact Registry.

Cloud Run desplegará la imagen en el puerto 8501 con acceso público, facturación
por solicitud, mínimo cero y máximo una instancia. El script admitirá `-DryRun`
para validar el plan sin crear o modificar recursos.

## Experimentos

El README principal documentará cómo ejecutar Optuna, comparar trials en MLflow
y abrir `optuna.db` mediante Optuna Dashboard. Los archivos generados permanecerán
ignorados por Git.

## Publicación

La rama inicial será `main`. Se creará el repositorio público
`hecgarri/wine-quality-cloud-run`, se revisará el conjunto exacto de archivos y se
hará un commit y push directo, sin pull request.
