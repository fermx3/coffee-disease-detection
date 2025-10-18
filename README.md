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

- make run_split_dataset  ← divide las imágenes de la carpeta raw_data en train/val/test de acuerdo a las carpetas que existen dentro de raw_data.
