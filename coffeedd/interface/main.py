
import numpy as np
from colorama import Fore, Style
from PIL import Image
from io import BytesIO
import base64

from coffeedd.ml_logic.data import create_dataset_from_directory, create_tf_dataset
from coffeedd.ml_logic.custom_weights import get_class_weights
from coffeedd.ml_logic.registry_ml import load_model, save_results, save_model, mlflow_transition_model, mlflow_run
from coffeedd.ml_logic.model import initialize_model, compile_model, train_model
from coffeedd.ml_logic.preprocessing import preprocess_image
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

def pred(img_source=None) -> dict:
    """
    Predicción flexible que acepta:
    - Ruta de archivo (str): '/path/to/image.jpg'
    - Bytes (bytes): contenido de imagen
    - Base64 (str): string base64 codificado
    """
    print(Fore.MAGENTA + "\n🔎 Empezando predicción... 🔎" + Style.RESET_ALL)

    model, _ = load_model()
    assert model is not None, "Modelo no cargado."

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
        "all_predictions": {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
    }
