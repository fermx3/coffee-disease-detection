
import numpy as np
from colorama import Fore, Style

from coffeedd.ml_logic.data import create_dataset_from_directory, create_tf_dataset
from coffeedd.ml_logic.custom_weights import get_class_weights
from coffeedd.ml_logic.registry_ml import load_model, save_results, save_model, mlflow_transition_model, mlflow_run
from coffeedd.ml_logic.model import initialize_model, compile_model, train_model
from coffeedd.params import LOCAL_DATA_PATH, CLASS_NAMES, SAMPLE_SIZE, BATCH_SIZE, FINE_TUNE, SAMPLE_NAME, MODEL_TARGET

@mlflow_run
def train():
    print(Fore.MAGENTA + "\n⭐️ Empezando el entrenamiento del modelo... ⭐️" + Style.RESET_ALL)

    print("🚀 Cargando datos...")
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
    model, useefficientnet = load_model()

    if model is None:
        model, useefficientnet = initialize_model(train_labels)

    model = compile_model(model)

    model, history = train_model(
        model,
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        class_weights=class_weights,
        use_efficientnet=useefficientnet,
        fine_tune=FINE_TUNE
    )

    val_recall = np.max(history.history['val_recall'])
    print(Fore.GREEN + f"\n✅ Entrenamiento completado. Mejor recall en validación: {val_recall:.4f}" + Style.RESET_ALL)

    val_disease_recall = np.max(history.history['val_disease_recall'])
    print(Fore.GREEN + f"\n✅ Entrenamiento completado. Mejor disease recall en validación: {val_disease_recall:.4f}" + Style.RESET_ALL)

    params = dict(
        context="train",
        training_set_size=SAMPLE_NAME,
        img_count=len(train_labels),
        useefficientnet=useefficientnet,
        fine_tune=FINE_TUNE
    )

    # Save results on the hard drive using coffeedd.ml_logic.registry
    save_results(params=params, metrics=dict(val_recall=val_recall, val_disease_recall=val_disease_recall))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # El ultimo modelo debe ser movido a "Staging" en MLflow si se usa MLflow
    if MODEL_TARGET == "mlflow":
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print(Fore.MAGENTA + "\n🏁 Proceso de entrenamiento finalizado. 🏁" + Style.RESET_ALL)

    return {"val_recall":val_recall, "val_disease_recall":val_disease_recall}
