# Coffee Disease Detection – Guía rápida

1) Instalar paquete en modo editable
- make reinstall_package

## Estructura de datos requerida

El proyecto espera la siguiente estructura de carpetas para los datos:

- data/
  - processed_data/    ← datos ya procesados (entradas listas para entrenamiento/evaluación)
  - raw_data/        ← datos originales (crudos)

Ejemplo en forma de árbol:

```
data/
├── processed_data/
└── raw_data/
```

Notas:
- Coloca las imágenes sin procesar en data/raw_data/.
- Guarda los resultados del preprocesado (por ejemplo: recortes, normalización, etiquetas procesadas) en data/processed_data/.
- Asegúrate de que los scripts que leen/guardan datos apunten a estas rutas relativas desde la raíz del proyecto.

## Funciones

- make run_split_dataset  ← divide las imágenes de la carpeta data/raw_data en train/val/test de acuerdo a las carpetas que existen dentro de data/raw_data.
- make run_split_resized_dataset ← divide las imágenes redimensionadas de la carpeta data/raw_data_224 en train/val/test de acuerdo a las carpetas que existen dentro de data/raw_data_224 para mantener las labels.
- make map_paths_and_labels ← Genera una lista de las rutas de las imágenes, etiquetas numéricas y nombres de clases desde un directorio estructurado por clases; al final imprime las clases detectadas y el total de imágenes.
- preprocess_raw_letterbox_224 ← Preprocesa imágenes desde un árbol de carpetas de clases a archivos de tamaño fijo (letterbox/recorte central/redimensionado), guardándolos en una carpeta de destino, preservando la estructura por clases, ejecutándose en paralelo y devolviendo un resumen.
