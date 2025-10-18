# Estructura de datos requerida

El proyecto espera la siguiente estructura de carpetas para los datos:

- data/
  - process_data/    ← datos ya procesados (entradas listas para entrenamiento/evaluación)
  - raw_data/        ← datos originales (crudos)

Ejemplo en forma de árbol:

```
data/
├── process_data/
└── raw_data/
```

Notas:
- Coloca las imágenes sin procesar en data/raw_data/.
- Guarda los resultados del preprocesado (por ejemplo: recortes, normalización, etiquetas procesadas) en data/process_data/.
- Asegúrate de que los scripts que leen/guardan datos apunten a estas rutas relativas desde la raíz del proyecto.
