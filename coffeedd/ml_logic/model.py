import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from coffeedd.params import *
from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric
from coffeedd.ml_logic.data_analysis import false_negatives_analysis

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, layers
from keras.applications import EfficientNetB0

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model(train_labels: list) -> Model:
    """Inicializa y devuelve el modelo adecuado según el tamaño del dataset.
    Args:
        input_shape (tuple): Forma de las imágenes de entrada (altura, anchura, canales).
    Returns:
        Model: Modelo Keras compilado listo para entrenar.
    """
    print(Fore.BLUE + "\n🏗️  Construyendo modelo..." + Style.RESET_ALL)
    # Decidir qué modelo usar según tamaño del dataset
    if len(train_labels) < 5000:
        print(f"📊 Dataset pequeño detectado ({len(train_labels)} imágenes)")
        print("🔧 Usando modelo CNN simple (más adecuado para pocos datos)")
        model = build_simple_cnn_model()
        use_efficientnet = False
        model_name = "CNN_simple"
    else:
        print(f"📊 Dataset grande detectado ({len(train_labels)} imágenes)")
        print("🔧 Usando EfficientNetB0 con transfer learning")
        model, base_model = build_efficientnet_model()
        use_efficientnet = True
        model_name = "EfficientNetB0"

    print("✅ Modelo inicializado")
    print(f"🏷️  Modelo seleccionado: {model_name}")

    return model, use_efficientnet

def build_simple_cnn_model():
    """Modelo CNN simple para datasets pequeños (< 5000 imágenes)"""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Data augmentation REALISTA para hojas de café
    x = layers.RandomFlip("horizontal")(inputs)  # Solo horizontal, NO vertical
    x = layers.RandomRotation(0.03)(x)  # ±30° (0.08 * 360° ≈ 29°)
    x = layers.RandomZoom(0.05)(x)  # Zoom moderado (reducido de 0.2)
    x = layers.RandomContrast(0.05)(x)  # Contraste variable (fotos de campo)

    # Bloque 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # Bloque 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # Bloque 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Bloque 4 (agregado para mejor detección)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier optimizado para recall
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    cnn_model = keras.Model(inputs, outputs)
    return cnn_model

def build_efficientnet_model():
    """Modelo EfficientNet para datasets grandes (>= 5000 imágenes)"""
    # Base model pre-entrenado
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Congelar base inicialmente
    base_model.trainable = False

    # Modelo completo SIN augmentation (se hace en el dataset)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model directamente
    x = base_model(inputs, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1024, activation='relu', name="dense_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, activation='relu', name="dense_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(256, activation='relu', name="dense_3")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs)

    return model, base_model

def compile_model(model: Model, learning_rate=LEARNING_RATE) -> Model:
    """Compila el modelo con el optimizador, la función de pérdida y las métricas adecuadas."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),
            keras.metrics.Precision(name='precision'),
            DiseaseRecallMetric(),  # Nueva métrica personalizada para recall por enfermedad
            keras.metrics.AUC(name='auc')
        ]
    )

    print("✅ Modelo compilado")
    print("\n📋 Resumen del modelo:")
    model.summary()

def train_model(
        model: Model,
        train_dataset,
        train_labels,
        val_dataset,
        val_labels,
        class_weights: dict,
        use_efficientnet: bool,
        fine_tune: bool = True
) -> Tuple[Model, dict]:
    """Entrena el modelo en dos fases:
    Fase 1: Entrenamiento con EfficientNet congelado (15 epochs)
    Args:
        model (Model): Modelo Keras a entrenar.
        train_dataset: Dataset de entrenamiento.
        val_dataset: Dataset de validación.
        val_labels: Etiquetas de validación (para métricas personalizadas).
        class_weights (dict): Pesos de clase para manejar el desbalance.
        use_efficientnet (bool): Indica si se está usando EfficientNet.
        fine_tune (bool): Indica si se debe realizar fine-tuning en fase 2.
    Returns:
        Tuple[Model, dict]: Modelo entrenado y el historial de entrenamiento.
    """
    checkpoint_filename = f'{MODELS_PATH}/best_model_{"EfficientNetB0" if use_efficientnet else "CNN"}_{len(val_labels)}.keras'

    class RecallFocusedCallback(keras.callbacks.Callback):
        """Callback personalizado para monitorear y reportar el recall de enfermedades cada 3 epochs."""
        def __init__(self, validation_data, class_names, val_labels):
            super().__init__()
            self.validation_data = validation_data
            self.class_names = class_names
            self.val_labels = val_labels
            self.best_disease_recall = 0.0

        def on_epoch_end(self, epoch, logs=None):
            """Al final de cada x epoch, calcular y mostrar el recall por enfermedad."""
            if (epoch + 1) % 3 == 0:  # Cada 3 epochs
                # Predecir en validación
                y_pred = self.model.predict(self.validation_data, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)

                # Calcular recall de ENFERMEDADES (excluyendo healthy)
                disease_recalls = []
                print(f"\n{'='*50}")
                print(f"📊 RECALL DETALLADO - Epoch {epoch + 1}")
                print(f"{'='*50}")

                for idx, class_name in enumerate(self.class_names):
                    mask = self.val_labels == idx
                    if np.sum(mask) > 0:
                        recall = np.sum(y_pred_classes[mask] == idx) / np.sum(mask)

                        if class_name != 'healthy':
                            disease_recalls.append(recall)

                        emoji = "🌱" if class_name == 'healthy' else "🦠"
                        print(f"  {emoji} {class_name:15s}: {recall:.4f}")

                # Desplegar cuantas muestras de enfermedad fueron clasificadas correctamente por enfermedad
                print("\n✅ Muestras de enfermedad clasificadas correctamente:")
                for idx, class_name in enumerate(self.class_names):
                    if class_name != 'healthy':
                        mask = self.val_labels == idx
                        total_should_be_tagged = np.sum(mask)  # Total que debería ser etiquetado
                        correct_count = np.sum(y_pred_classes[mask] == idx)  # Correctamente clasificado
                        percentage = (correct_count / total_should_be_tagged) * 100 if total_should_be_tagged > 0 else 0
                        print(f"  🦠 {class_name:15s}: {correct_count} / {total_should_be_tagged} - ({percentage:.2f}%)")



                # Recall promedio de enfermedades
                avg_disease_recall = np.mean(disease_recalls) if disease_recalls else 0
                print(f"\n🎯 Recall promedio enfermedades: {avg_disease_recall:.4f}")

                if avg_disease_recall > self.best_disease_recall:
                    self.best_disease_recall = avg_disease_recall
                    print("✨ ¡Nuevo mejor recall de enfermedades!")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_recall',  # Mantener recall como monitor principal
            patience=15,  # Usar más paciencia para datasets pequeños
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',
            factor=0.3,
            patience=5,
            min_lr=1e-8,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_filename,
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        RecallFocusedCallback(val_dataset, CLASS_NAMES, val_labels)
    ]

    print("\n" + "="*60)
    if use_efficientnet:
        print("🚂 FASE 1: Entrenando con EfficientNet congelado (15 epochs)")
    else:
        print("🚂 ENTRENAMIENTO: Modelo CNN simple (30 epochs)")
    print("="*60)

    # Ajustar epochs según tipo de modelo
    initial_epochs = 15 if use_efficientnet else 30

    history_phase1 = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fase 1 de entrenamiento completada")
    print(f"📈Recall máximo en validación durante fase 1: {max(history_phase1.history['val_recall']):.4f}")

    # Solo hacer fine-tuning si usamos EfficientNet Y tenemos suficientes datos Y la fase 1 fue exitosa
    should_finetune = (
        use_efficientnet and
        len(train_labels) >= 10000 and
        len(history_phase1.history['val_accuracy']) > 0 and
        max(history_phase1.history['val_accuracy']) > 0.60 and  # Umbral mínimo de accuracy
        fine_tune
    )

    if should_finetune:
        model, history_phase2 = fine_tune_model(
            model,
            model.layers[1],  # base_model está en la segunda posición
            history_phase1,
            train_dataset,
            val_dataset,
            class_weights,
            callbacks
        )
    else:
        if not use_efficientnet:
            print("\n" + "="*60)
            print("ℹ️  Modelo CNN simple: No requiere fine-tuning")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"⚠️  FASE 2: Saltando fine-tuning (dataset pequeño: {len(train_labels)} imágenes)")
            print("="*60)
            print("ℹ️  Se requieren al menos 10,000 imágenes para fine-tuning seguro.")
            print("ℹ️  El modelo se mantiene con la base congelada (solo la cabeza entrenada).")

        # Crear un history_phase2 vacío para evitar errores
        history_phase2 = history_phase1

    return model, history_phase1, history_phase2

def fine_tune_model(
        model: Model,
        base_model,
        history_phase1,
        train_dataset,
        val_dataset,
        class_weights: dict,
        callbacks
        ):
    """Realiza fine-tuning del modelo descongelando las últimas capas de EfficientNet.
    Args:
        model (Model): Modelo Keras a entrenar.
        base_model: La base EfficientNet del modelo.
        history_phase1: Historial del entrenamiento previo.
        train_dataset: Dataset de entrenamiento.
        val_dataset: Dataset de validación.
        class_weights (dict): Pesos de clase para manejar el desbalance.
        callbacks: Callbacks para el entrenamiento.
    Returns:
        Tuple[Model, dict]: Modelo entrenado y el historial de entrenamiento.
    """
    print("\n" + "="*60)
    print("🔥 FASE 2: Fine-tuning (descongelando últimas 15 capas)")
    print("="*60)
    print(f"ℹ️  Mejor val_accuracy en Fase 1: {max(history_phase1.history['val_accuracy']):.3f}")

    # Descongelar base model
    base_model.trainable = True

    # Congelar MÁS capas (solo descongelar las últimas 15 en lugar de 30)
    fine_tune_at = len(base_model.layers) - 15  # Cambio de 30 a 15
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Capas entrenables: {sum([1 for l in model.layers if l.trainable])}")
    print(f"Capas totales: {len(model.layers)}")

    # Recompilar con learning rate MUY bajo (10x más bajo que antes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Recall(name='recall'),  # Recall general
            keras.metrics.Precision(name='precision'),
            DiseaseRecallMetric(),  # Recall específico de enfermedades
            keras.metrics.AUC(name='auc')
        ]
    )


    history_phase2 = model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=len(history_phase1.epoch),
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return model, history_phase2

def evaluate_model(
        model: Model,
        test_dataset,
        test_labels,
        useefficientnet: bool,
        get_confusion_matrix: bool = False,
        get_false_negatives_analysis: bool = True
    ) -> Tuple[Model, dict]:
    """
    Evalúa el modelo en un dataset dado.
    Args:
        model (Model): Modelo Keras a evaluar.
        test_dataset: Dataset de test.
        test_labels: Etiquetas reales del test set.
        useefficientnet (bool): Indica si se está usando EfficientNet.
    Returns:
        Tuple[Model, dict]: Resultados de la evaluación.
    """

    if model is None:
        print(Fore.RED + "❌ Modelo no está definido. No se puede evaluar." + Style.RESET_ALL)
        return None

    model_name = "EfficientNetB0" if useefficientnet else "CNN"

    print("\n" + "="*60)
    print("🧪 EVALUACIÓN FINAL EN TEST SET")
    print("="*60)

    # Evaluar en test
    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

    # Predecir en test
    y_pred_test = model.predict(test_dataset, verbose=1)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)

    # Identificar clases presentes en test
    unique_test_classes = np.unique(test_labels)
    print(f"\nClases presentes en test set: {unique_test_classes}")

    # Verificar si todas las clases están presentes
    missing_classes = set(range(NUM_CLASSES)) - set(unique_test_classes)
    if missing_classes:
        print(f"⚠️  Clases ausentes en test: {[CLASS_NAMES[i] for i in missing_classes]}")

    # Classification report
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)

    # Opción: Especificar solo las clases presentes (recomendado para muestras pequeñas)
    if len(unique_test_classes) < NUM_CLASSES:
        # Usar solo nombres de clases presentes
        target_names_present = [CLASS_NAMES[i] for i in unique_test_classes]
        print(classification_report(test_labels, y_pred_test_classes,
                                labels=unique_test_classes,
                                target_names=target_names_present,
                                digits=4))
        print(f"\n⚠️  Nota: Solo se muestran las {len(unique_test_classes)} clases presentes en el test set.")
    else:
        # Todas las clases presentes, usar reporte completo
        print(classification_report(test_labels, y_pred_test_classes,
                                target_names=CLASS_NAMES,
                                digits=4))

    if get_confusion_matrix:
        # Matriz de confusión
        cm = confusion_matrix(test_labels, y_pred_test_classes, labels=unique_test_classes)

        # Usar solo nombres de clases presentes para los ejes
        axis_labels = [CLASS_NAMES[i] for i in unique_test_classes]

        # Nombre descriptivo para la matriz de confusión
        confusion_matrix_filename = f'{MODELS_PATH}/confusion_matrix_{model_name}_{SAMPLE_NAME}.png'

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=axis_labels, yticklabels=axis_labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Matriz de Confusión - Test Set\n{model_name} - {SAMPLE_NAME}',
                fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(confusion_matrix_filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"💾 Matriz de confusión guardada: {confusion_matrix_filename}")

    if get_false_negatives_analysis:
        # Análisis detallado de falsos negativos
        false_negatives_analysis(
            test_labels,
            y_pred_test_classes
        )

    return test_results
