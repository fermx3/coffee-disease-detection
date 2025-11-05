# Coffee Disease Detection 🌱🔬

**Sistema de detección de enfermedades en plantas de café usando Computer Vision**

---

## 🚀 **Inicio Rápido con Modelo Pre-entrenado**

### **1. Obtener el Código**
```bash
# Si no tienes el repositorio
git clone https://github.com/fermx3/coffee-disease-detection.git
cd coffee-disease-detection
```

### **2. Setup del Entorno**
```bash
# Crear estructura de carpetas necesaria
make reset_local_files

# Instalar dependencias y paquete
make reinstall_package
```

### **3. Descargar y Colocar Modelo Pre-entrenado**

**Descargar modelo desde:** [Enlace a Google Drive o fuente pública]

**Colocar según el tipo de modelo:**

```bash
# Para modelo VGG16:
cp modelo_descargado.keras ~/.coffeedd/mlops/training_outputs/models/vgg16/

# Para modelo EfficientNet:
cp modelo_descargado.keras ~/.coffeedd/mlops/training_outputs/models/efficientnet/
```

**Estructura esperada:**
```
~/.coffeedd/mlops/
└── training_outputs/
    └── models/
        ├── vgg16/          ← Colocar modelos VGG16 aquí
        └── efficientnet/   ← Colocar modelos EfficientNet aquí
```

### **4. Ejecutar API**
```bash
# Iniciar servidor de API
make run_api

# API disponible en: http://localhost:8000/docs
```

### **5. Probar el Sistema**
- Abrir http://localhost:8000/docs en tu navegador
- Usar el endpoint `POST /predict` para subir una imagen
- Ver resultados de predicción en tiempo real

---

## 🔧 **Comandos Adicionales**

### **Entrenamiento y Evaluación**
```bash
make run_train              # Entrenar nuevo modelo
make run_evaluate           # Evaluar modelo existente
make run_pred               # Predicción individual
```

### **Procesamiento de Datos**
```bash
make run_split_dataset      # Dividir dataset en train/val/test
make map_paths_and_labels   # Mapear rutas y etiquetas
make preprocess_raw_letterbox_224  # Preprocesar imágenes a 224x224
```

### **Calidad de Código**
```bash
make test                   # Ejecutar tests
make format                 # Formatear código
make install                # Instalar dependencias
```

---

## 📁 **Estructura de Datos**

El proyecto espera la siguiente estructura para datos de entrenamiento:

```
data/
├── processed_data/    ← Datos listos para entrenamiento
│   ├── healthy/      ← Hojas sanas
│   ├── cerscospora/  ← Enfermedad Cercospora
│   ├── leaf_rust/    ← Roya del café
│   ├── miner/        ← Minador de hoja
│   └── phoma/        ← Enfermedad Phoma
└── raw_data/         ← Datos originales sin procesar
```

---

## 🚨 **Troubleshooting**

**Problema: "No model found"**
```bash
# Verificar que el modelo esté en la carpeta correcta
ls ~/.coffeedd/mlops/training_outputs/models/vgg16/
ls ~/.coffeedd/mlops/training_outputs/models/efficientnet/
```

**Problema: "Module not found"**
```bash
# Reinstalar paquete
make reinstall_package
```

**Problema: API no inicia**
```bash
# Verificar dependencias
make install
```

---

## 📊 **Tipos de Modelo Soportados**

| Modelo | Descripción | Carpeta de Destino |
|--------|-------------|-------------------|
| **VGG16** | Transfer learning estable | `~/.coffeedd/mlops/training_outputs/models/vgg16/` |
| **EfficientNet** | Modelo optimizado | `~/.coffeedd/mlops/training_outputs/models/efficientnet/` |

---

## 🔗 **Enlaces Útiles**

- **API Documentation**: http://localhost:8000/docs (cuando API esté corriendo)
- **Redoc**: http://localhost:8000/redoc
- **Test Endpoint**: http://localhost:8000/

---

**Última actualización**: Noviembre 2025
