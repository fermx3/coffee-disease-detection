
import numpy as np
from colorama import Fore, Style
from PIL import Image
from io import BytesIO
import base64
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from coffeedd.ml_logic.data import create_dataset_from_directory, create_tf_dataset
from coffeedd.ml_logic.custom_weights import get_class_weights
from coffeedd.ml_logic.registry_ml import load_model, save_results, save_model, mlflow_transition_model, mlflow_run
from coffeedd.ml_logic.model import initialize_model, compile_model, train_model
from coffeedd.ml_logic.data_analysis import plot_training_metrics_combined, analyze_training_convergence_combined, analyze_false_negatives, analyze_disease_recall
from coffeedd.ml_logic.gcs_upload import upload_latest_model_to_gcs, list_models_in_gcs
from coffeedd.params import LOCAL_DATA_PATH, CLASS_NAMES, SAMPLE_SIZE, BATCH_SIZE, FINE_TUNE, SAMPLE_NAME, MODEL_TARGET, MODELS_PATH, NUM_CLASSES, MODEL_NAME, PRODUCTION_MODEL

# Sistema de caché global para el modelo
_cached_model = None
_cached_model_path = None
_production_model_cache = None  # Caché específico para modelo de producción

def get_cached_model():
    """
    Obtiene el modelo desde caché o lo carga si no está en memoria.

    Estrategia de carga:
    1. Si PRODUCTION_MODEL está definido, usa ese modelo específico (rápido)
    2. Si no, busca el modelo más reciente (normal)
    3. Para GCS: evita búsquedas innecesarias cuando hay modelo en producción

    Returns:
        Model: Modelo Keras cargado o None si no existe
    """
    global _cached_model, _cached_model_path, _production_model_cache

    try:
        # Verificar si hay un modelo específico de producción configurado
        production_model = PRODUCTION_MODEL

        if production_model:
            return _get_production_model(production_model)
        else:
            return _get_latest_model()

    except Exception as e:
        print(Fore.RED + f"❌ Error en get_cached_model(): {e}" + Style.RESET_ALL)
        # En caso de error, intentar carga directa
        return load_model()

def _get_production_model(production_model_name):
    """
    Carga un modelo específico de producción (rápido, sin búsquedas)

    Args:
        production_model_name: Nombre del modelo (ej: model_VGG16_20251102-073551.keras)
    """
    global _production_model_cache

    from coffeedd.params import LOCAL_REGISTRY_PATH, MODEL_ARCHITECTURE
    import os

    # Si ya tenemos el modelo de producción en caché
    if _production_model_cache is not None and _production_model_cache.get('name') == production_model_name:
        print(Fore.GREEN + f"⚡ Usando modelo de producción desde caché: {production_model_name}" + Style.RESET_ALL)
        return _production_model_cache['model']

    # Buscar el modelo específico localmente
    models_base_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    architecture_dir = os.path.join(models_base_dir, MODEL_ARCHITECTURE.lower())
    production_model_path = os.path.join(architecture_dir, production_model_name)

    # Si existe localmente, usarlo
    if os.path.exists(production_model_path):
        print(Fore.CYAN + f"🎯 Cargando modelo de producción: {production_model_name}" + Style.RESET_ALL)
        print(Fore.BLUE + f"📁 Desde: {production_model_path}" + Style.RESET_ALL)

        # Cargar directamente sin buscar otros modelos
        from coffeedd.ml_logic.registry_ml import load_specific_model
        model = load_specific_model(production_model_path)

        if model:
            # Almacenar en caché de producción
            _production_model_cache = {
                'name': production_model_name,
                'path': production_model_path,
                'model': model
            }
            print(Fore.GREEN + f"✅ Modelo de producción cargado y cacheado" + Style.RESET_ALL)
            return model

    # Si no existe localmente y MODEL_TARGET=gcs, intentar descargarlo
    if MODEL_TARGET == "gcs":
        print(Fore.YELLOW + f"⚠️  Modelo de producción no encontrado localmente: {production_model_name}" + Style.RESET_ALL)
        print(Fore.BLUE + f"🔍 Buscando en GCS..." + Style.RESET_ALL)

        # Buscar específicamente este modelo en GCS
        model = _download_specific_model_from_gcs(production_model_name)
        if model:
            _production_model_cache = {
                'name': production_model_name,
                'path': f"gcs:{production_model_name}",
                'model': model
            }
            return model

    # Si no se encuentra el modelo de producción, fallback al método normal
    print(Fore.YELLOW + f"⚠️  Modelo de producción '{production_model_name}' no encontrado" + Style.RESET_ALL)
    print(Fore.BLUE + f"🔄 Fallback: buscando último modelo disponible..." + Style.RESET_ALL)
    return _get_latest_model()

def _get_latest_model():
    """
    Obtiene el modelo más reciente (método original)
    """
    global _cached_model, _cached_model_path

    from coffeedd.ml_logic.registry_ml import find_latest_model_by_architecture
    from coffeedd.params import LOCAL_REGISTRY_PATH, MODEL_ARCHITECTURE
    import os

    models_base_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    latest_model_path = find_latest_model_by_architecture(models_base_dir, MODEL_ARCHITECTURE.lower())

    # Si no hay modelo en disco, verificar GCS si es el target
    if latest_model_path is None and MODEL_TARGET == "gcs":
        print(Fore.BLUE + "🔍 Verificando modelo en GCS..." + Style.RESET_ALL)
        # Forzar carga desde GCS (esto actualizará el caché local)
        _cached_model = load_model()
        _cached_model_path = "gcs_loaded"  # Marcar como cargado desde GCS
        return _cached_model

    # Si no hay modelo disponible
    if latest_model_path is None:
        print(Fore.YELLOW + "⚠️  No hay modelo disponible" + Style.RESET_ALL)
        _cached_model = None
        _cached_model_path = None
        return None

    # Si ya tenemos el modelo en caché y la ruta no ha cambiado
    if _cached_model is not None and _cached_model_path == latest_model_path:
        print(Fore.GREEN + "⚡ Usando modelo desde caché en memoria" + Style.RESET_ALL)
        return _cached_model

    # Si hay un modelo más reciente o no tenemos caché
    if _cached_model_path != latest_model_path:
        if _cached_model_path is not None:
            print(Fore.BLUE + "🔄 Modelo actualizado detectado, recargando caché..." + Style.RESET_ALL)
        else:
            print(Fore.BLUE + "🔄 Cargando modelo en caché..." + Style.RESET_ALL)

        # Cargar el modelo
        _cached_model = load_model()
        _cached_model_path = latest_model_path

        if _cached_model is not None:
            print(Fore.GREEN + "✅ Modelo cargado y almacenado en caché" + Style.RESET_ALL)

        return _cached_model

    return _cached_model

def _download_specific_model_from_gcs(model_name):
    """
    Descarga un modelo específico desde GCS sin buscar todos los modelos

    Args:
        model_name: Nombre del modelo a descargar
    """
    try:
        from google.cloud import storage
        from coffeedd.params import BUCKET_NAME, LOCAL_REGISTRY_PATH, MODEL_ARCHITECTURE
        import os

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Construir la ruta en GCS
        gcs_path = f"models/{MODEL_ARCHITECTURE.lower()}/{model_name}"
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            print(Fore.RED + f"❌ Modelo no encontrado en GCS: {gcs_path}" + Style.RESET_ALL)
            return None

        # Crear directorio local
        architecture_dir = os.path.join(LOCAL_REGISTRY_PATH, "models", MODEL_ARCHITECTURE.lower())
        os.makedirs(architecture_dir, exist_ok=True)

        # Descargar
        local_path = os.path.join(architecture_dir, model_name)
        print(Fore.BLUE + f"📥 Descargando modelo específico: {gcs_path}" + Style.RESET_ALL)
        blob.download_to_filename(local_path)

        # Cargar el modelo descargado
        from coffeedd.ml_logic.registry_ml import load_specific_model
        model = load_specific_model(local_path)

        if model:
            print(Fore.GREEN + f"✅ Modelo de producción descargado y cargado" + Style.RESET_ALL)

        return model

    except Exception as e:
        print(Fore.RED + f"❌ Error descargando modelo específico: {e}" + Style.RESET_ALL)
        return None

def clear_model_cache():
    """Limpia el caché del modelo (útil después de entrenar un nuevo modelo)"""
    global _cached_model, _cached_model_path
    _cached_model = None
    _cached_model_path = None
    print(Fore.BLUE + "🧹 Caché del modelo limpiado" + Style.RESET_ALL)

def warm_model_cache():
    """Precalienta el caché cargando el modelo en memoria"""
    print(Fore.CYAN + "🔥 Precalentando caché del modelo..." + Style.RESET_ALL)
    model = get_cached_model()
    if model is not None:
        print(Fore.GREEN + "✅ Caché precalentado exitosamente" + Style.RESET_ALL)
        return True
    else:
        print(Fore.YELLOW + "⚠️  No hay modelo disponible para precalentar" + Style.RESET_ALL)
        return False

def get_cache_status():
    """Obtiene el estado actual del caché"""
    global _cached_model, _cached_model_path

    model_architecture = None
    if _cached_model:
        try:
            from coffeedd.ml_logic.registry_ml import detect_model_architecture
            model_architecture = detect_model_architecture(_cached_model.layers)
        except:
            model_architecture = "unknown"

    status = {
        "has_cached_model": _cached_model is not None,
        "cached_model_path": _cached_model_path,
        "model_layers": len(_cached_model.layers) if _cached_model else None,
        "model_architecture": model_architecture
    }

    return status

@mlflow_run
def train(metrics_viz=True, test_mode=False):
    print(Fore.MAGENTA + "\n⭐️ Empezando el entrenamiento del modelo... ⭐️" + Style.RESET_ALL)

    print(f"🚀 Cargando datos... con {SAMPLE_NAME}")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        create_dataset_from_directory(LOCAL_DATA_PATH, CLASS_NAMES, sample_size=SAMPLE_SIZE)

    # Crear datasets de TensorFlow
    train_dataset = create_tf_dataset(train_paths, train_labels, BATCH_SIZE,
                                    is_training=True, augment=True)
    val_dataset = create_tf_dataset(val_paths, val_labels, BATCH_SIZE,
                                    is_training=False, augment=False)
    test_dataset = create_tf_dataset(test_paths, test_labels, BATCH_SIZE,
                                    is_training=False, augment=False)

    print("\n✅ Datasets creados exitosamente (carga on-the-fly activada)")

    class_weights = get_class_weights(
        train_labels=train_labels,
    )

    #Entrenar el modelo usando `model.py`
        # Cargar o inicializar modelo
    try:
        model = get_cached_model()
        if model is not None:
            print(Fore.GREEN + "✅ Modelo existente cargado exitosamente" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "⚠️  No hay modelo existente, creando uno nuevo..." + Style.RESET_ALL)
            model = initialize_model(train_labels)
    except Exception as e:
        print(Fore.YELLOW + f"⚠️  Error al cargar modelo existente: {str(e)[:100]}" + Style.RESET_ALL)
        print(Fore.BLUE + "🔄 Creando modelo nuevo..." + Style.RESET_ALL)
        model = initialize_model(train_labels)

    model = compile_model(model)

    model, history = train_model(
        model,
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        class_weights=class_weights,
        fine_tune=FINE_TUNE
    )

    val_recall = np.max(history.history['val_recall'])
    print(Fore.GREEN + f"\n✅ Entrenamiento completado. Mejor recall en validación: {val_recall:.4f}" + Style.RESET_ALL)

    val_disease_recall = np.max(history.history['val_disease_recall'])
    print(Fore.GREEN + f"\n✅ Entrenamiento completado. Mejor disease recall en validación: {val_disease_recall:.4f}" + Style.RESET_ALL)

    # ==========================================
    # NUEVO: VISUALIZACIONES DE MÉTRICAS DE ENTRENAMIENTO (HISTORIAL COMBINADO)
    # ==========================================
    if metrics_viz:
        training_viz_metrics = {}
        convergence_metrics = {}

        if history is not None:
            print(f"\n{Fore.MAGENTA}📊 Generando visualizaciones de entrenamiento...{Style.RESET_ALL}")

            model_name = MODEL_NAME

            # Generar visualizaciones de métricas de entrenamiento
            training_viz_metrics = plot_training_metrics_combined(
                combined_history=history,
                model_name=model_name,
                sample_name=SAMPLE_NAME,
                test_labels=test_labels,
                y_pred_test_classes=None,
                verbose=True
            )

            # Análisis de convergencia del entrenamiento
            convergence_metrics = analyze_training_convergence_combined(
                combined_history=history,
                verbose=True
            )
            print(f"     - Visualización entrenamiento: {len(training_viz_metrics)}")
            print(f"     - Convergencia: {len(convergence_metrics)}")

            # Información sobre archivos generados
            files_generated = []

            if training_viz_metrics:
                training_metrics_file = f'{MODELS_PATH}/training_metrics_{model_name}.png'
                files_generated.append(training_metrics_file)

            print(f"   • Archivos generados: {len(files_generated)}")
            for i, file_path in enumerate(files_generated, 1):
                print(f"     {i}. {file_path}")


        else:
            print(f"\n{Fore.YELLOW}⚠️  No se proporcionó historial de entrenamiento. Saltando visualizaciones.{Style.RESET_ALL}")


    params = dict(
        context="train" if not test_mode else "test_train",
        training_set_size=SAMPLE_NAME,
        img_count=len(train_labels),
        model_name=model_name,
        fine_tune=FINE_TUNE
    )

    # Guardar resultados y modelo entrenado si no es modo test
    if not test_mode:
        # Combinar métricas manejando duplicados (convergence_metrics tiene prioridad)
        combined_metrics = training_viz_metrics.copy()
        combined_metrics.update(convergence_metrics)

        # Save results on the hard drive using coffeedd.ml_logic.registry
        save_results(params=params, metrics=combined_metrics)

        # Save model weight on the hard drive (and optionally on GCS too!)
        save_model(model=model)

        # Limpiar caché después de guardar nuevo modelo
        clear_model_cache()

        # El ultimo modelo debe ser movido a "Staging" en MLflow si se usa MLflow
        if MODEL_TARGET == "mlflow":
            mlflow_transition_model(current_stage="None", new_stage="Staging")

    print(Fore.MAGENTA + "\n🏁 Proceso de entrenamiento finalizado. 🏁" + Style.RESET_ALL)

    return {"val_recall":val_recall, "val_disease_recall":val_disease_recall}

@mlflow_run
def evaluate(confusion_matrix_viz=True, false_negatives_analysis=True):
    """
    Evalúa el modelo entrenado en el test set y genera métricas detalladas

    Args:
        combined_history: History combinado de train_model() (opcional)
    """
    print(Fore.MAGENTA + "\n🧪 Empezando evaluación del modelo... 🧪" + Style.RESET_ALL)

    # Cargar el modelo entrenado
    model = get_cached_model()
    assert model is not None, "Modelo no encontrado. Entrena el modelo primero."

    print(f"🚀 Cargando datos de test... con {SAMPLE_NAME}")

    # Cargar datasets (solo necesitamos test)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        create_dataset_from_directory(LOCAL_DATA_PATH, CLASS_NAMES, sample_size=SAMPLE_SIZE)

    # Crear dataset de test
    test_dataset = create_tf_dataset(test_paths, test_labels, BATCH_SIZE,
                                   is_training=False, augment=False)

    print("\n✅ Dataset de test creado exitosamente")

    # Constantes
    model_name = MODEL_NAME
    sample_name = SAMPLE_NAME

    print("\n" + "="*60)
    print(Fore.CYAN + "🧪 EVALUACIÓN FINAL EN TEST SET" + Style.RESET_ALL)
    print("="*60)

    # Evaluar en test
    print(Fore.YELLOW + "📊 Calculando métricas en test set..." + Style.RESET_ALL)
    test_results = model.evaluate(test_dataset, verbose=1)

    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_recall = test_results[3] if len(test_results) > 3 else 0.0

    print(f"\n{Fore.GREEN}📈 Métricas del Test Set:{Style.RESET_ALL}")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Test Accuracy: {test_accuracy:.4f}")
    print(f"   • Test Recall: {test_recall:.4f}")

    # Predecir en test
    print(f"\n{Fore.YELLOW}🔮 Generando predicciones...{Style.RESET_ALL}")
    y_pred_test = model.predict(test_dataset, verbose=1)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)

    # Identificar clases presentes en test
    unique_test_classes = np.unique(test_labels)
    print(f"\n{Fore.BLUE}🏷️  Clases presentes en test set: {unique_test_classes}{Style.RESET_ALL}")

    # Verificar si todas las clases están presentes
    missing_classes = set(range(NUM_CLASSES)) - set(unique_test_classes)
    if missing_classes:
        missing_class_names = [CLASS_NAMES[i] for i in missing_classes]
        print(f"{Fore.YELLOW}⚠️  Clases ausentes en test: {missing_class_names}{Style.RESET_ALL}")

    # Classification report
    print("\n" + "="*60)
    print(Fore.CYAN + "📊 CLASSIFICATION REPORT" + Style.RESET_ALL)
    print("="*60)

    # Especificar solo las clases presentes (recomendado para muestras pequeñas)
    if len(unique_test_classes) < NUM_CLASSES:
        # Usar solo nombres de clases presentes
        target_names_present = [CLASS_NAMES[i] for i in unique_test_classes]
        classification_rep = classification_report(
            test_labels, y_pred_test_classes,
            labels=unique_test_classes,
            target_names=target_names_present,
            digits=4
        )
        print(classification_rep)
        print(f"\n{Fore.YELLOW}⚠️  Nota: Solo se muestran las {len(unique_test_classes)} clases presentes en el test set.{Style.RESET_ALL}")
    else:
        # Todas las clases presentes, usar reporte completo
        classification_rep = classification_report(
            test_labels, y_pred_test_classes,
            target_names=CLASS_NAMES,
            digits=4
        )
        print(classification_rep)

    if confusion_matrix_viz:
        # Matriz de confusión
        print(f"\n{Fore.YELLOW}📈 Generando matriz de confusión...{Style.RESET_ALL}")
        cm = confusion_matrix(test_labels, y_pred_test_classes, labels=unique_test_classes)

        # Usar solo nombres de clases presentes para los ejes
        axis_labels = [CLASS_NAMES[i] for i in unique_test_classes]

        # Crear directorio de modelos si no existe
        os.makedirs(MODELS_PATH, exist_ok=True)

        # Nombre descriptivo para la matriz de confusión
        confusion_matrix_filename = f'{MODELS_PATH}/confusion_matrix_{model_name}_{sample_name}.png'

        # Crear la matriz de confusión
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=axis_labels, yticklabels=axis_labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Matriz de Confusión - Test Set\n{model_name} - {sample_name}',
                fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(confusion_matrix_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria

        print(f"{Fore.GREEN}💾 Matriz de confusión guardada: {confusion_matrix_filename}{Style.RESET_ALL}")

    # ==========================================
    # ANÁLISIS DE FALSOS NEGATIVOS
    # ==========================================
    fn_metrics = {}
    disease_metrics = {}

    if false_negatives_analysis:
        print(f"\n{Fore.MAGENTA}🔍 Ejecutando análisis de falsos negativos...{Style.RESET_ALL}")

        # Análisis de falsos negativos por clase
        fn_metrics = analyze_false_negatives(
            test_labels=test_labels,
            y_pred_test_classes=y_pred_test_classes,
            verbose=True
        )

        # Análisis de disease recall (detección binaria de enfermedades)
        disease_metrics = analyze_disease_recall(
            test_labels=test_labels,
            y_pred_test_classes=y_pred_test_classes,
            verbose=True
        )

    # ==========================================
    # GUARDAR RESULTADOS EN MLFLOW
    # ==========================================

    # Parámetros base
    params = dict(
        context="evaluate",
        training_set_size=SAMPLE_NAME,
        test_img_count=len(test_labels),
        model_name=model_name,
        classes_in_test=len(unique_test_classes),
        total_classes=NUM_CLASSES,
    )

    # Métricas base del modelo
    base_metrics = dict(
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        test_recall=test_recall
    )

    # Combinar TODAS las métricas para MLflow (solo las que existen)
    all_metrics = {
        **base_metrics,               # Métricas básicas de evaluación
        **fn_metrics,                # Análisis de falsos negativos (vacío si no se ejecutó)
        **disease_metrics,           # Métricas de disease recall (vacío si no se ejecutó)
    }

    print(f"\n{Fore.CYAN}📤 Subiendo métricas a MLflow...{Style.RESET_ALL}")
    print(f"   • Parámetros: {len(params)} items")
    print(f"   • Métricas totales: {len(all_metrics)} items")
    print(f"     - Evaluación básica: {len(base_metrics)}")
    if fn_metrics:
        print(f"     - Falsos negativos: {len(fn_metrics)}")
    if disease_metrics:
        print(f"     - Disease detection: {len(disease_metrics)}")

    # Guardar resultados (esto incluirá TODAS las métricas en MLflow)
    save_results(params=params, metrics=all_metrics)

    print("\n" + "="*60)
    print(Fore.GREEN + "✅ EVALUACIÓN COMPLETADA EXITOSAMENTE" + Style.RESET_ALL)
    print("="*60)
    print(f"{Fore.GREEN}📋 Resumen:{Style.RESET_ALL}")
    print(f"   • Imágenes evaluadas: {len(test_labels)}")
    print(f"   • Clases evaluadas: {len(unique_test_classes)}/{NUM_CLASSES}")
    print(f"   • Accuracy final: {test_accuracy:.4f}")
    print(f"   • Recall final: {test_recall:.4f}")

    # Solo mostrar métricas si se calcularon
    if disease_metrics and 'disease_recall' in disease_metrics:
        print(f"   • Disease Recall: {disease_metrics['disease_recall']:.4f}")
    if fn_metrics and 'total_false_negatives' in fn_metrics:
        print(f"   • Falsos Negativos totales: {fn_metrics['total_false_negatives']}")
        print(f"   • Tasa FN global: {fn_metrics.get('overall_false_negative_rate', 0):.1f}%")

    # Información sobre archivos generados
    files_generated = [confusion_matrix_filename]

    print(f"   • Archivos generados: {len(files_generated)}")
    for i, file_path in enumerate(files_generated, 1):
        print(f"     {i}. {file_path}")

    print(f"   • Total métricas en MLflow: {len(all_metrics)}")

    # Retornar métricas combinadas
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "disease_recall": disease_metrics.get('disease_recall', None),
        "classes_evaluated": len(unique_test_classes),
        "total_images": len(test_labels),
        "confusion_matrix_path": confusion_matrix_filename,
        "total_false_negatives": fn_metrics.get('total_false_negatives', None),
        "false_negative_rate": fn_metrics.get('overall_false_negative_rate', None),
        "mlflow_metrics_count": len(all_metrics),
        "files_generated": files_generated,
    }

def pred(img_source=None) -> dict:
    """
    Predicción flexible que acepta:
    - Ruta de archivo (str): '/path/to/image.jpg'
    - Bytes (bytes): contenido de imagen
    - Base64 (str): string base64 codificado
    """
    print(Fore.MAGENTA + "\n🔎 Empezando predicción... 🔎" + Style.RESET_ALL)

    # Intentar cargar modelo existente
    model = get_cached_model()

    if model is None:
        print(Fore.YELLOW + "⚠️  No se pudo cargar modelo existente" + Style.RESET_ALL)
        print(Fore.BLUE + "🔄 Para hacer predicciones, necesitas entrenar un modelo primero" + Style.RESET_ALL)
        print(Fore.CYAN + "💡 Ejecuta: make run_train" + Style.RESET_ALL)
        raise ValueError("No hay modelo disponible para predicción. Entrena un modelo primero con 'make run_train'")

    print(Fore.GREEN + "✅ Modelo cargado exitosamente" + Style.RESET_ALL)

    if img_source is None:
        img_source = input("Ingresa la ruta de la imagen: ").strip()

    # MANEJO ROBUSTO DE DIFERENTES INPUTS
    try:
        if isinstance(img_source, bytes):
            # Caso 1: Bytes directos (desde UploadFile)
            img = Image.open(BytesIO(img_source))

        elif isinstance(img_source, str):
            # Caso 2: String - puede ser ruta O base64

            # Detectar si es base64
            if img_source.startswith('data:image'):
                # Formato: data:image/jpeg;base64,/9j/4AA...
                img_source = img_source.split(',')[1]

            # Intentar decodificar base64
            try:
                img_data = base64.b64decode(img_source)
                img = Image.open(BytesIO(img_data))
            except:
                # Si falla, es una ruta de archivo
                img = Image.open(img_source)
        else:
            raise ValueError(f"Tipo de input no soportado: {type(img_source)}")

    except Exception as e:
        raise ValueError(f"Error al cargar imagen: {str(e)}")

    # Preprocesar
    img = img.resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # Predecir
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    probability = float(np.max(predictions))

    print(Fore.GREEN + f"\n✅ Predicción: {CLASS_NAMES[predicted_class]} ({probability:.4f})" + Style.RESET_ALL)

    return {
        "class_name": CLASS_NAMES[predicted_class],
        "probability": probability,
        "raw": {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
    }

@mlflow_run
def upload_model_to_gcs(model_version: str = None, dry_run: bool = False):
    """
    Sube el último modelo entrenado a Google Cloud Storage

    Args:
        model_version: Versión específica del modelo (si None, usa timestamp)
        dry_run: Si es True, solo simula la subida sin ejecutarla

    Returns:
        dict: Información sobre la subida
    """
    print(Fore.MAGENTA + "\n☁️  Subiendo modelo a GCS... ☁️" + Style.RESET_ALL)

    try:
        # Ejecutar subida
        result = upload_latest_model_to_gcs(
            model_version=model_version,
            include_metadata=True,
            dry_run=dry_run
        )

        if not dry_run:
            # Guardar información de la subida en MLflow
            save_results(
                params={
                    "context": "gcs_upload",
                    "model_version": result["model_version"],
                    "gcs_bucket": "configured",
                    "include_metadata": True
                },
                metrics={
                    "model_size_mb": result["model_size_mb"],
                    "metadata_fields": result["metadata_fields"],
                    "upload_success": 1 if result["success"] else 0
                }
            )

            print(f"\n{Fore.GREEN}✅ Modelo subido exitosamente a GCS{Style.RESET_ALL}")
            print(f"   • Versión: {result['model_version']}")
            print(f"   • Tamaño: {result['model_size_mb']:.2f} MB")
            print(f"   • Ruta GCS: {result['gcs_paths']['model']}")

        return result

    except Exception as e:
        print(f"{Fore.RED}❌ Error subiendo modelo: {e}{Style.RESET_ALL}")
        raise

def list_gcs_models(limit: int = 10):
    """
    Lista los modelos disponibles en GCS

    Args:
        limit: Número máximo de modelos a mostrar
    """
    print(Fore.CYAN + "\n📋 Modelos en Google Cloud Storage" + Style.RESET_ALL)
    print("="*60)

    try:
        models = list_models_in_gcs(limit=limit)

        if not models:
            print(f"{Fore.YELLOW}📂 No se encontraron modelos en GCS{Style.RESET_ALL}")
            return []

        print(f"\n📊 Encontrados {len(models)} modelos:")
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   📏 Tamaño: {model['size_mb']:.2f} MB")
            print(f"   📅 Creado: {model['created'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   🔗 Ruta: {model['gcs_path']}")

        return models

    except Exception as e:
        print(f"{Fore.RED}❌ Error listando modelos: {e}{Style.RESET_ALL}")
        return []
