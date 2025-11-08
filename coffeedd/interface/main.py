"""Main interface for Coffee Disease Detection ML operations:
- Training
- Evaluation
- Prediction
"""

from io import BytesIO
import base64
import os

import numpy as np
from colorama import Fore, Style
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from coffeedd.ml_logic.data import create_dataset_from_directory, create_tf_dataset
from coffeedd.ml_logic.custom_weights import get_class_weights
from coffeedd.ml_logic.registry_ml import (
    load_model,
    save_results,
    save_model,
    mlflow_transition_model,
    mlflow_run,
)
from coffeedd.ml_logic.model import initialize_model, compile_model, train_model
from coffeedd.ml_logic.data_analysis import (
    plot_training_metrics_combined,
    analyze_training_convergence_combined,
    analyze_false_negatives,
    analyze_disease_recall,
)
from coffeedd.ml_logic.gcs_upload import upload_latest_model_to_gcs, list_models_in_gcs
from coffeedd.params import (
    LOCAL_DATA_PATH,
    CLASS_NAMES,
    SAMPLE_SIZE,
    BATCH_SIZE,
    FINE_TUNE,
    SAMPLE_NAME,
    MODEL_TARGET,
    MODELS_PATH,
    NUM_CLASSES,
    MODEL_NAME,
    PRODUCTION_MODEL,
    LOCAL_REGISTRY_PATH,
    MODEL_ARCHITECTURE,
)

# Global model cache system
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None
_PRODUCTION_MODEL_CACHE = None


def get_cached_model():
    """Get the model from cache or load it if not in memory.

    Loading strategy:
    1. If PRODUCTION_MODEL is defined, use that specific model (fast)
    2. Otherwise, search for the most recent model (normal)
    3. For GCS: avoid unnecessary searches when there's a production model

    Returns:
        Loaded Keras model or None if it doesn't exist.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _PRODUCTION_MODEL_CACHE

    try:
        # Verificar si hay un modelo específico de producción configurado
        production_model = PRODUCTION_MODEL

        if production_model:
            return _get_production_model(production_model)
        else:
            return _get_latest_model()

    except Exception as e:
        print(Fore.RED + f"❌ Error en get_cached_model(): {e}" + Style.RESET_ALL)
        # In case of error, attempt direct loading
        return load_model()


def _get_production_model(production_model_name):
    """Load a specific production model (fast, no searches).

    Args:
        production_model_name: Model name (e.g., model_VGG16_20251102-073551.keras)
    """
    global _PRODUCTION_MODEL_CACHE

    # If we already have the production model in cache
    if (
        _PRODUCTION_MODEL_CACHE is not None
        and _PRODUCTION_MODEL_CACHE.get("name") == production_model_name
    ):
        print(
            Fore.GREEN
            + f"⚡ Usando modelo de producción desde caché: {production_model_name}"
            + Style.RESET_ALL
        )
        return _PRODUCTION_MODEL_CACHE["model"]

    # Search for the specific model locally
    models_base_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    architecture_dir = os.path.join(models_base_dir, MODEL_ARCHITECTURE.lower())
    production_model_path = os.path.join(architecture_dir, production_model_name)

    # If it exists locally, use it
    if os.path.exists(production_model_path):
        print(
            Fore.CYAN
            + f"🎯 Cargando modelo de producción: {production_model_name}"
            + Style.RESET_ALL
        )
        print(Fore.BLUE + f"📁 Desde: {production_model_path}" + Style.RESET_ALL)

        # Load directly without searching for other models
        from coffeedd.ml_logic.registry_ml import load_specific_model

        model = load_specific_model(production_model_path)

        if model:
            # Store in production cache
            _PRODUCTION_MODEL_CACHE = {
                "name": production_model_name,
                "path": production_model_path,
                "model": model,
            }
            print(
                Fore.GREEN
                + "✅ Modelo de producción cargado y cacheado"
                + Style.RESET_ALL
            )
            return model

    # If it doesn't exist locally and MODEL_TARGET=gcs, try to download it
    if MODEL_TARGET == "gcs":
        print(
            Fore.YELLOW
            + f"⚠️  Modelo de producción no encontrado localmente: {production_model_name}"
            + Style.RESET_ALL
        )
        print(Fore.BLUE + "🔍 Buscando en GCS..." + Style.RESET_ALL)

        # Search specifically for this model in GCS
        model = _download_specific_model_from_gcs(production_model_name)
        if model:
            _PRODUCTION_MODEL_CACHE = {
                "name": production_model_name,
                "path": f"gcs:{production_model_name}",
                "model": model,
            }
            return model

    # If production model not found, fallback to normal method
    print(
        Fore.YELLOW
        + f"⚠️  Modelo de producción '{production_model_name}' no encontrado"
        + Style.RESET_ALL
    )
    print(
        Fore.BLUE
        + "🔄 Fallback: buscando último modelo disponible..."
        + Style.RESET_ALL
    )
    return _get_latest_model()


def _get_latest_model():
    """Get the most recent model (original method)."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    from coffeedd.ml_logic.registry_ml import find_latest_model_by_architecture
    from coffeedd.params import LOCAL_REGISTRY_PATH, MODEL_ARCHITECTURE
    import os

    models_base_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    latest_model_path = find_latest_model_by_architecture(
        models_base_dir, MODEL_ARCHITECTURE.lower()
    )

    # If no model on disk, check GCS if it's the target
    if latest_model_path is None and MODEL_TARGET == "gcs":
        print(Fore.BLUE + "🔍 Verificando modelo en GCS..." + Style.RESET_ALL)
        # Force load from GCS (this will update local cache)
        _CACHED_MODEL = load_model()
        _CACHED_MODEL_PATH = "gcs_loaded"  # Mark as loaded from GCS
        return _CACHED_MODEL

    # If no model available
    if latest_model_path is None:
        print(Fore.YELLOW + "⚠️  No hay modelo disponible" + Style.RESET_ALL)
        _CACHED_MODEL = None
        _CACHED_MODEL_PATH = None
        return None

    # If we already have the model in cache and path hasn't changed
    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == latest_model_path:
        print(Fore.GREEN + "⚡ Usando modelo desde caché en memoria" + Style.RESET_ALL)
        return _CACHED_MODEL

    # If there's a newer model or we don't have cache
    if _CACHED_MODEL_PATH != latest_model_path:
        if _CACHED_MODEL_PATH is not None:
            print(
                Fore.BLUE
                + "🔄 Modelo actualizado detectado, recargando caché..."
                + Style.RESET_ALL
            )
        else:
            print(Fore.BLUE + "🔄 Cargando modelo en caché..." + Style.RESET_ALL)

        # Load the model
        _CACHED_MODEL = load_model()
        _CACHED_MODEL_PATH = latest_model_path

        if _CACHED_MODEL is not None:
            print(
                Fore.GREEN + "✅ Modelo cargado y almacenado en caché" + Style.RESET_ALL
            )

        return _CACHED_MODEL

    return _CACHED_MODEL


def _download_specific_model_from_gcs(model_name):
    """Download a specific model from GCS without searching all models.

    Args:
        model_name: Name of the model to download.
    """
    try:
        from google.cloud import storage
        from coffeedd.params import BUCKET_NAME, LOCAL_REGISTRY_PATH, MODEL_ARCHITECTURE
        import os

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Build GCS path
        gcs_path = f"models/{MODEL_ARCHITECTURE.lower()}/{model_name}"
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            print(
                Fore.RED
                + f"❌ Modelo no encontrado en GCS: {gcs_path}"
                + Style.RESET_ALL
            )
            return None

        # Create local directory
        architecture_dir = os.path.join(
            LOCAL_REGISTRY_PATH, "models", MODEL_ARCHITECTURE.lower()
        )
        os.makedirs(architecture_dir, exist_ok=True)

        # Download
        local_path = os.path.join(architecture_dir, model_name)
        print(
            Fore.BLUE
            + f"📥 Descargando modelo específico: {gcs_path}"
            + Style.RESET_ALL
        )
        blob.download_to_filename(local_path)

        # Load the downloaded model
        from coffeedd.ml_logic.registry_ml import load_specific_model

        model = load_specific_model(local_path)

        if model:
            print(
                Fore.GREEN
                + "✅ Modelo de producción descargado y cargado"
                + Style.RESET_ALL
            )

        return model

    except Exception as e:
        print(
            Fore.RED + f"❌ Error descargando modelo específico: {e}" + Style.RESET_ALL
        )
        return None


def clear_model_cache():
    """Clear the model cache (useful after training a new model)."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    _CACHED_MODEL = None
    _CACHED_MODEL_PATH = None
    print(Fore.BLUE + "🧹 Caché del modelo limpiado" + Style.RESET_ALL)


def warm_model_cache():
    """Preload the model cache by loading the model into memory."""
    print(Fore.CYAN + "🔥 Precalentando caché del modelo..." + Style.RESET_ALL)
    model = get_cached_model()
    if model is not None:
        print(Fore.GREEN + "✅ Caché precalentado exitosamente" + Style.RESET_ALL)
        return True
    else:
        print(
            Fore.YELLOW
            + "⚠️  No hay modelo disponible para precalentar"
            + Style.RESET_ALL
        )
        return False


def get_cache_status():
    """Get the current cache status."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    model_architecture = None
    if _CACHED_MODEL:
        try:
            from coffeedd.ml_logic.registry_ml import detect_model_architecture

            model_architecture = detect_model_architecture(_CACHED_MODEL.layers)
        except:
            model_architecture = "unknown"

    status = {
        "has_cached_model": _CACHED_MODEL is not None,
        "cached_model_path": _CACHED_MODEL_PATH,
        "model_layers": len(_CACHED_MODEL.layers) if _CACHED_MODEL else None,
        "model_architecture": model_architecture,
    }

    return status


@mlflow_run
def train(metrics_viz=True, test_mode=False):
    print(
        Fore.MAGENTA
        + "\n⭐️ Empezando el entrenamiento del modelo... ⭐️"
        + Style.RESET_ALL
    )

    print(f"🚀 Cargando datos... con {SAMPLE_NAME}")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        create_dataset_from_directory(
            LOCAL_DATA_PATH, CLASS_NAMES, sample_size=SAMPLE_SIZE
        )
    )

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(
        train_paths, train_labels, BATCH_SIZE, is_training=True, augment=True
    )
    val_dataset = create_tf_dataset(
        val_paths, val_labels, BATCH_SIZE, is_training=False, augment=False
    )
    test_dataset = create_tf_dataset(
        test_paths, test_labels, BATCH_SIZE, is_training=False, augment=False
    )

    print("\n✅ Datasets creados exitosamente (carga on-the-fly activada)")

    class_weights = get_class_weights(
        train_labels=train_labels,
    )

    # Train the model using `model.py`
    # Load or initialize model
    try:
        model = get_cached_model()
        if model is not None:
            print(
                Fore.GREEN
                + "✅ Modelo existente cargado exitosamente"
                + Style.RESET_ALL
            )
        else:
            print(
                Fore.YELLOW
                + "⚠️  No hay modelo existente, creando uno nuevo..."
                + Style.RESET_ALL
            )
            model = initialize_model()
    except Exception as e:
        print(
            Fore.YELLOW
            + f"⚠️  Error al cargar modelo existente: {str(e)[:100]}"
            + Style.RESET_ALL
        )
        print(Fore.BLUE + "🔄 Creando modelo nuevo..." + Style.RESET_ALL)
        model = initialize_model()

    model = compile_model(model)

    model, history = train_model(
        model,
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        class_weights=class_weights,
        fine_tune=FINE_TUNE,
    )

    val_recall = np.max(history.history["val_recall"])
    print(
        Fore.GREEN
        + f"\n✅ Entrenamiento completado. Mejor recall en validación: {val_recall:.4f}"
        + Style.RESET_ALL
    )

    val_disease_recall = np.max(history.history["val_disease_recall"])
    print(
        Fore.GREEN
        + f"\n✅ Entrenamiento completado. Mejor disease recall en validación: {val_disease_recall:.4f}"
        + Style.RESET_ALL
    )

    # Training metrics visualizations (combined history)
    if metrics_viz:
        training_viz_metrics = {}
        convergence_metrics = {}

        if history is not None:
            print(
                f"\n{Fore.MAGENTA}📊 Generando visualizaciones de entrenamiento...{Style.RESET_ALL}"
            )

            model_name = MODEL_NAME

            # Generate training metrics visualizations
            training_viz_metrics = plot_training_metrics_combined(
                combined_history=history,
                model_name=model_name,
                sample_name=SAMPLE_NAME,
                test_labels=test_labels,
                y_pred_test_classes=None,
                verbose=True,
            )

            # Training convergence analysis
            convergence_metrics = analyze_training_convergence_combined(
                combined_history=history, verbose=True
            )
            print(f"     - Visualización entrenamiento: {len(training_viz_metrics)}")
            print(f"     - Convergencia: {len(convergence_metrics)}")

            # Information about generated files
            files_generated = []

            if training_viz_metrics:
                training_metrics_file = (
                    f"{MODELS_PATH}/training_metrics_{model_name}.png"
                )
                files_generated.append(training_metrics_file)

            print(f"   • Archivos generados: {len(files_generated)}")
            for i, file_path in enumerate(files_generated, 1):
                print(f"     {i}. {file_path}")

        else:
            print(
                f"\n{Fore.YELLOW}⚠️  No se proporcionó historial de entrenamiento. Saltando visualizaciones.{Style.RESET_ALL}"
            )

    params = dict(
        context="train" if not test_mode else "test_train",
        training_set_size=SAMPLE_NAME,
        img_count=len(train_labels),
        model_name=model_name,
        fine_tune=FINE_TUNE,
    )

    # Save results and trained model if not in test mode
    if not test_mode:
        # Combine metrics handling duplicates (convergence_metrics takes priority)
        combined_metrics = training_viz_metrics.copy()
        combined_metrics.update(convergence_metrics)

        # Save results on the hard drive using coffeedd.ml_logic.registry
        save_results(params=params, metrics=combined_metrics)

        # Save model weight on the hard drive (and optionally on GCS too!)
        save_model(model=model)

        # Clear cache after saving new model
        clear_model_cache()

        # The latest model should be moved to "Staging" in MLflow if MLflow is used
        if MODEL_TARGET == "mlflow":
            mlflow_transition_model(current_stage="None", new_stage="Staging")

    print(
        Fore.MAGENTA + "\n🏁 Proceso de entrenamiento finalizado. 🏁" + Style.RESET_ALL
    )

    return {"val_recall": val_recall, "val_disease_recall": val_disease_recall}


@mlflow_run
def evaluate(confusion_matrix_viz=True, false_negatives_analysis=True):
    """Evaluate the trained model on test set and generate detailed metrics.

    Args:
        confusion_matrix_viz: Whether to generate confusion matrix visualization.
        false_negatives_analysis: Whether to perform false negatives analysis.
    """
    print(Fore.MAGENTA + "\n🧪 Empezando evaluación del modelo... 🧪" + Style.RESET_ALL)

    # Load the trained model
    model = get_cached_model()
    assert model is not None, "Modelo no encontrado. Entrena el modelo primero."

    print(f"🚀 Cargando datos de test... con {SAMPLE_NAME}")

    # Load datasets (we only need test)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        create_dataset_from_directory(
            LOCAL_DATA_PATH, CLASS_NAMES, sample_size=SAMPLE_SIZE
        )
    )

    # Create test dataset
    test_dataset = create_tf_dataset(
        test_paths, test_labels, BATCH_SIZE, is_training=False, augment=False
    )

    print("\n✅ Dataset de test creado exitosamente")

    # Constants
    model_name = MODEL_NAME
    sample_name = SAMPLE_NAME

    print("\n" + "=" * 60)
    print(Fore.CYAN + "🧪 EVALUACIÓN FINAL EN TEST SET" + Style.RESET_ALL)
    print("=" * 60)

    # Evaluate on test
    print(Fore.YELLOW + "📊 Calculando métricas en test set..." + Style.RESET_ALL)
    test_results = model.evaluate(test_dataset, verbose=1)

    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_recall = test_results[3] if len(test_results) > 3 else 0.0

    print(f"\n{Fore.GREEN}📈 Métricas del Test Set:{Style.RESET_ALL}")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Test Accuracy: {test_accuracy:.4f}")
    print(f"   • Test Recall: {test_recall:.4f}")

    # Predict on test
    print(f"\n{Fore.YELLOW}🔮 Generando predicciones...{Style.RESET_ALL}")
    y_pred_test = model.predict(test_dataset, verbose=1)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)

    # Identify classes present in test
    unique_test_classes = np.unique(test_labels)
    print(
        f"\n{Fore.BLUE}🏷️  Clases presentes en test set: {unique_test_classes}{Style.RESET_ALL}"
    )

    # Check if all classes are present
    missing_classes = set(range(NUM_CLASSES)) - set(unique_test_classes)
    if missing_classes:
        missing_class_names = [CLASS_NAMES[i] for i in missing_classes]
        print(
            f"{Fore.YELLOW}⚠️  Clases ausentes en test: {missing_class_names}{Style.RESET_ALL}"
        )

    # Classification report
    print("\n" + "=" * 60)
    print(Fore.CYAN + "📊 CLASSIFICATION REPORT" + Style.RESET_ALL)
    print("=" * 60)

    # Specify only present classes (recommended for small samples)
    if len(unique_test_classes) < NUM_CLASSES:
        # Use only names of present classes
        target_names_present = [CLASS_NAMES[i] for i in unique_test_classes]
        classification_rep = classification_report(
            test_labels,
            y_pred_test_classes,
            labels=unique_test_classes,
            target_names=target_names_present,
            digits=4,
        )
        print(classification_rep)
        print(
            f"\n{Fore.YELLOW}⚠️  Nota: Solo se muestran las {len(unique_test_classes)} clases presentes en el test set.{Style.RESET_ALL}"
        )
    else:
        # All classes present, use complete report
        classification_rep = classification_report(
            test_labels, y_pred_test_classes, target_names=CLASS_NAMES, digits=4
        )
        print(classification_rep)

    if confusion_matrix_viz:
        # Confusion matrix
        print(f"\n{Fore.YELLOW}📈 Generando matriz de confusión...{Style.RESET_ALL}")
        cm = confusion_matrix(
            test_labels, y_pred_test_classes, labels=unique_test_classes
        )

        # Use only names of present classes for axes
        axis_labels = [CLASS_NAMES[i] for i in unique_test_classes]

        # Create models directory if it doesn't exist
        os.makedirs(MODELS_PATH, exist_ok=True)

        # Descriptive name for confusion matrix
        confusion_matrix_filename = f"{MODELS_PATH}/confusion_matrix_{model_name}.png"

        # Create the confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=axis_labels,
            yticklabels=axis_labels,
            cbar_kws={"label": "Count"},
        )
        plt.title(
            f"Matriz de Confusión - Test Set\n{model_name} - {sample_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Etiqueta Real", fontsize=12)
        plt.xlabel("Predicción", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(confusion_matrix_filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close figure to free memory

        print(
            f"{Fore.GREEN}💾 Matriz de confusión guardada: {confusion_matrix_filename}{Style.RESET_ALL}"
        )

    # False negatives analysis
    fn_metrics = {}
    disease_metrics = {}

    if false_negatives_analysis:
        print(
            f"\n{Fore.MAGENTA}🔍 Ejecutando análisis de falsos negativos...{Style.RESET_ALL}"
        )

        # False negatives analysis by class
        fn_metrics = analyze_false_negatives(
            test_labels=test_labels,
            y_pred_test_classes=y_pred_test_classes,
            verbose=True,
        )

        # Disease recall analysis (binary disease detection)
        disease_metrics = analyze_disease_recall(
            test_labels=test_labels,
            y_pred_test_classes=y_pred_test_classes,
            verbose=True,
        )

    # Save results to MLflow

    # Base parameters
    params = dict(
        context="evaluate",
        training_set_size=SAMPLE_NAME,
        test_img_count=len(test_labels),
        model_name=model_name,
        classes_in_test=len(unique_test_classes),
        total_classes=NUM_CLASSES,
    )

    # Base model metrics
    base_metrics = dict(
        test_loss=test_loss, test_accuracy=test_accuracy, test_recall=test_recall
    )

    # Combine ALL metrics for MLflow (only existing ones)
    all_metrics = {
        **base_metrics,  # Basic evaluation metrics
        **fn_metrics,  # False negatives analysis (empty if not executed)
        **disease_metrics,  # Disease recall metrics (empty if not executed)
    }

    print(f"\n{Fore.CYAN}📤 Subiendo métricas a MLflow...{Style.RESET_ALL}")
    print(f"   • Parámetros: {len(params)} items")
    print(f"   • Métricas totales: {len(all_metrics)} items")
    print(f"     - Evaluación básica: {len(base_metrics)}")
    if fn_metrics:
        print(f"     - Falsos negativos: {len(fn_metrics)}")
    if disease_metrics:
        print(f"     - Disease detection: {len(disease_metrics)}")

    # Save results (this will include ALL metrics in MLflow)
    save_results(params=params, metrics=all_metrics)

    print("\n" + "=" * 60)
    print(Fore.GREEN + "✅ EVALUACIÓN COMPLETADA EXITOSAMENTE" + Style.RESET_ALL)
    print("=" * 60)
    print(f"{Fore.GREEN}📋 Resumen:{Style.RESET_ALL}")
    print(f"   • Imágenes evaluadas: {len(test_labels)}")
    print(f"   • Clases evaluadas: {len(unique_test_classes)}/{NUM_CLASSES}")
    print(f"   • Accuracy final: {test_accuracy:.4f}")
    print(f"   • Recall final: {test_recall:.4f}")

    # Only show metrics if they were calculated
    if disease_metrics and "disease_recall" in disease_metrics:
        print(f"   • Disease Recall: {disease_metrics['disease_recall']:.4f}")
    if fn_metrics and "total_false_negatives" in fn_metrics:
        print(f"   • Falsos Negativos totales: {fn_metrics['total_false_negatives']}")
        print(
            f"   • Tasa FN global: {fn_metrics.get('overall_false_negative_rate', 0):.1f}%"
        )

    # Information about generated files
    files_generated = [confusion_matrix_filename]

    print(f"   • Archivos generados: {len(files_generated)}")
    for i, file_path in enumerate(files_generated, 1):
        print(f"     {i}. {file_path}")

    print(f"   • Total métricas en MLflow: {len(all_metrics)}")

    # Return combined metrics
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_recall": test_recall,
        "disease_recall": disease_metrics.get("disease_recall", None),
        "classes_evaluated": len(unique_test_classes),
        "total_images": len(test_labels),
        "confusion_matrix_path": confusion_matrix_filename,
        "total_false_negatives": fn_metrics.get("total_false_negatives", None),
        "false_negative_rate": fn_metrics.get("overall_false_negative_rate", None),
        "mlflow_metrics_count": len(all_metrics),
        "files_generated": files_generated,
    }


def pred(img_source=None) -> dict:
    """Flexible prediction that accepts:
    - File path (str): '/path/to/image.jpg'
    - Bytes (bytes): image content
    - Base64 (str): base64 encoded string
    """
    print(Fore.MAGENTA + "\n🔎 Empezando predicción... 🔎" + Style.RESET_ALL)

    # Try to load existing model
    model = get_cached_model()

    if model is None:
        print(Fore.YELLOW + "⚠️  No se pudo cargar modelo existente" + Style.RESET_ALL)
        print(
            Fore.BLUE
            + "🔄 Para hacer predicciones, necesitas entrenar un modelo primero"
            + Style.RESET_ALL
        )
        print(Fore.CYAN + "💡 Ejecuta: make run_train" + Style.RESET_ALL)
        raise ValueError(
            "No hay modelo disponible para predicción. Entrena un modelo primero con 'make run_train'"
        )

    print(Fore.GREEN + "✅ Modelo cargado exitosamente" + Style.RESET_ALL)

    if img_source is None:
        img_source = input("Ingresa la ruta de la imagen: ").strip()

    # Robust handling of different inputs
    try:
        if isinstance(img_source, bytes):
            # Case 1: Direct bytes (from UploadFile)
            img = Image.open(BytesIO(img_source))

        elif isinstance(img_source, str):
            # Case 2: String - could be path OR base64

            # Detect if it's base64
            if img_source.startswith("data:image"):
                # Format: data:image/jpeg;base64,/9j/4AA...
                img_source = img_source.split(",")[1]

            # Try to decode base64
            try:
                img_data = base64.b64decode(img_source)
                img = Image.open(BytesIO(img_data))
            except:
                # If it fails, it's a file path
                img = Image.open(img_source)
        else:
            raise ValueError(f"Tipo de input no soportado: {type(img_source)}")

    except Exception as e:
        raise ValueError(f"Error al cargar imagen: {str(e)}")

    # Preprocess
    img = img.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    probability = float(np.max(predictions))

    print(
        Fore.GREEN
        + f"\n✅ Predicción: {CLASS_NAMES[predicted_class]} ({probability:.4f})"
        + Style.RESET_ALL
    )

    return {
        "class_name": CLASS_NAMES[predicted_class],
        "probability": probability,
        "raw": {
            CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
        },
    }


@mlflow_run
def upload_model_to_gcs(model_version: str = None, dry_run: bool = False):
    """Upload the latest trained model to Google Cloud Storage.

    Args:
        model_version: Specific model version (if None, uses timestamp).
        dry_run: If True, only simulates the upload without executing it.

    Returns:
        Information about the upload.
    """
    print(Fore.MAGENTA + "\n☁️  Subiendo modelo a GCS... ☁️" + Style.RESET_ALL)

    try:
        # Execute upload
        result = upload_latest_model_to_gcs(
            model_version=model_version, include_metadata=True, dry_run=dry_run
        )

        if not dry_run:
            # Save upload information to MLflow
            save_results(
                params={
                    "context": "gcs_upload",
                    "model_version": result["model_version"],
                    "gcs_bucket": "configured",
                    "include_metadata": True,
                },
                metrics={
                    "model_size_mb": result["model_size_mb"],
                    "metadata_fields": result["metadata_fields"],
                    "upload_success": 1 if result["success"] else 0,
                },
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
    """List available models in GCS.

    Args:
        limit: Maximum number of models to show.
    """
    print(Fore.CYAN + "\n📋 Modelos en Google Cloud Storage" + Style.RESET_ALL)
    print("=" * 60)

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
