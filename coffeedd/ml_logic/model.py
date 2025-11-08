"""Model initialization, compilation, and training logic for Coffee Disease Detection."""

import time
from typing import Tuple

from tensorflow import keras
from keras import Model, layers
from keras.applications import EfficientNetB0
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.regularizers import l2

from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from coffeedd.params import (
    IMG_SIZE,
    SAMPLE_SIZE,
    MODEL_ARCHITECTURE,
    NUM_CLASSES,
    MODEL_NAME,
    LEARNING_RATE,
    LOCAL_REGISTRY_PATH,
    SAMPLE_NAME,
    CLASS_NAMES,
    EPOCHS,
    MODELS_PATH,
)
from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric
from coffeedd.ml_logic.data_analysis import false_negatives_analysis
from coffeedd.utilities.results import combine_histories
from coffeedd.utilities.params_helpers import auto_type

print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()


end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


def initialize_model() -> Model:
    """Initialize and return the appropriate model based on dataset size and selected architecture.

    Args:
        train_labels: List of training labels.

    Returns:
        Compiled Keras model ready for training.
    """
    print(Fore.BLUE + "\n🏗️  Construyendo modelo..." + Style.RESET_ALL)

    model_architecture = MODEL_ARCHITECTURE.lower()

    if model_architecture == "cnn":
        print("🔧 Usando modelo CNN simple")
        model = build_simple_cnn_model()
        model_name = "CNN_simple"
    elif model_architecture == "vgg16":
        print("🔧 Usando VGG16 con transfer learning")
        model = build_vgg16_model()
        model_name = "VGG16"
    elif model_architecture == "efficientnet":
        print("🔧 Usando EfficientNetB0 con transfer learning")
        model, _ = build_efficientnet_model()
        model_name = "EfficientNetB0"
    else:
        raise ValueError(f"Arquitectura de modelo no soportada: {model_architecture}")

    print("✅ Modelo inicializado")
    print(f"🏷️  Modelo seleccionado: {model_name}")

    return model


def build_simple_cnn_model():
    """Build a simple CNN model for small datasets (< 5000 images)."""
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Realistic data augmentation for coffee leaves
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.03)(x)
    x = layers.RandomZoom(0.05)(x)
    x = layers.RandomContrast(0.05)(x)

    # Convolutional block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # Convolutional block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # Convolutional block 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Convolutional block 4
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier optimized for recall
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    cnn_model = keras.Model(inputs, outputs)
    return cnn_model


def build_vgg16_model():
    """Build VGG16 model with anti-overfitting optimizations for production generalization.

    Based on successful configuration with "half" dataset that demonstrated
    better production performance. Focus on:
    1. Excellent healthy vs diseased detection
    2. Better discrimination between specific diseases
    3. Preventing overfitting with full dataset
    """

    sample_size_typed = auto_type(SAMPLE_SIZE)

    if sample_size_typed is None or sample_size_typed == "full":
        is_large_dataset = True
        config_name = "DATASET GRANDE (59K+) - ANTI-OVERFITTING"
    elif sample_size_typed == "half":
        is_large_dataset = True
        config_name = "DATASET GRANDE (half=30K) - ANTI-OVERFITTING"
    elif isinstance(sample_size_typed, (int, float)):
        if isinstance(sample_size_typed, float) and 0 < sample_size_typed <= 1:
            # Percentage value
            estimated_size = int(sample_size_typed * 59807)
        else:
            # Absolute number of images - consider large only if > 10K
            estimated_size = int(sample_size_typed)

        # More realistic threshold: datasets < 10K are SMALL
        is_large_dataset = estimated_size >= 10000
        config_name = (
            f"DATASET {'GRANDE' if is_large_dataset else 'PEQUEÑO'} (~{estimated_size})"
        )
    else:
        is_large_dataset = False
        config_name = "DATASET PEQUEÑO"

    print(f"🎯 VGG16 Anti-Overfitting para {config_name}")

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    if is_large_dataset:
        # Anti-overfitting configuration for large datasets
        print("🛡️  Aplicando configuración ANTI-OVERFITTING para dataset grande...")

        # Ultra-conservative transfer learning - freeze even more layers
        for i, layer in enumerate(base_model.layers):
            if i < 17:  # Freeze 17/19 layers (more conservative than 15)
                layer.trainable = False
            else:  # Only train last 2 conv layers
                layer.trainable = True

        # Head inspired by successful "half" version but more conservative
        model = Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                # First layer: smaller than successful "half" version
                layers.Dense(
                    128,
                    activation="relu",  # Reduced from 256 to 128
                    kernel_regularizer=keras.regularizers.l2(0.003),
                ),  # Stronger L2 than 0.001
                layers.BatchNormalization(),
                layers.Dropout(0.5),  # More aggressive dropout than 0.4
                # Second layer: intermediate size for disease discrimination
                layers.Dense(
                    64,
                    activation="relu",  # Reduced from 128 to 64
                    kernel_regularizer=keras.regularizers.l2(0.003),
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.4),  # More aggressive dropout than 0.3
                # Third layer: NEW - for better disease discrimination
                layers.Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.002),
                ),  # Gentle L2
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                # Final layer
                layers.Dense(5, activation="softmax"),
            ]
        )

        print("   🔒 Capas congeladas: 17/19 (ultra-conservador)")
        print("   🧠 Head: 128→64→32→5 (anti-overfitting + discriminación)")
        print("   🛡️  L2: 0.003→0.003→0.002 (regularización fuerte)")
        print("   💧 Dropout: 0.5→0.4→0.3 (muy agresivo)")

    else:
        # Successful original configuration for small/medium dataset
        print("⚙️  Aplicando configuración EXITOSA para dataset pequeño/mediano...")

        for i, layer in enumerate(base_model.layers):
            if i < 15:  # Successful original configuration
                layer.trainable = False
            else:
                layer.trainable = True

        model = Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                # Configuration that worked well with "half"
                layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.001),
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(0.001),
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(5, activation="softmax"),
            ]
        )

        print("   🔒 Capas congeladas: 15/19 (configuración exitosa)")
        print("   🧠 Head: 256→128→5 (probado exitosamente)")
        print("   🛡️  L2: 0.001 (suave)")
        print("   💧 Dropout: 0.4→0.3 (moderado)")

    return model


def build_efficientnet_model():
    """Build EfficientNet model balanced for performance without sacrificing stability.

    Version 2.0 - Balanced adjustments:
    - Gentler L2 regularization (0.005/0.003 vs 0.01/0.005)
    - Less aggressive dropout (0.2 → 0.3 → 0.4 vs 0.3 → 0.4 → 0.5)
    - Larger dense layers (768 → 384 → 192 vs 512 → 256 → 128)
    - Standard BatchNormalization momentum (0.99 vs 0.9)
    - Better stability/performance balance
    """
    print("🔧 Construyendo EfficientNet BALANCEADO (V2.0)...")

    # Pre-trained base model
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base initially
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Base model
    x = base_model(inputs, training=False)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # BatchNorm with standard momentum (less conservative)
    x = layers.BatchNormalization(momentum=0.99, name="bn_1")(x)

    # First dense layer - INCREASED from 512 to 768
    x = layers.Dropout(0.2, name="dropout_1")(x)  # Dropout REDUCED from 0.3 to 0.2
    x = layers.Dense(
        768,  # INCREASED from 512 to 768
        activation="relu",
        kernel_regularizer=l2(0.005),  # L2 REDUCED from 0.01 to 0.005
        name="dense_1",
    )(x)

    x = layers.BatchNormalization(momentum=0.99, name="bn_2")(x)

    # Second dense layer - INCREASED from 256 to 384
    x = layers.Dropout(0.3, name="dropout_2")(x)  # Dropout REDUCED from 0.4 to 0.3
    x = layers.Dense(
        384,  # INCREASED from 256 to 384
        activation="relu",
        kernel_regularizer=l2(0.005),  # L2 same
        name="dense_2",
    )(x)

    x = layers.BatchNormalization(momentum=0.99, name="bn_3")(x)

    # Third dense layer - INCREASED from 128 to 192
    x = layers.Dropout(0.4, name="dropout_3")(x)  # Dropout REDUCED from 0.5 to 0.4
    x = layers.Dense(
        192,  # INCREASED from 128 to 192
        activation="relu",
        kernel_regularizer=l2(0.003),  # L2 GENTLER from 0.005 to 0.003
        name="dense_3",
    )(x)

    # Final dropout REDUCED
    x = layers.Dropout(0.2, name="dropout_final")(x)  # REDUCED from 0.3 to 0.2

    # Classification layer
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="balanced_efficientnet")

    print("✅ EfficientNet balanceado V2.0 construido:")
    print("   📊 Parámetros totales: ~{model.count_params():,}")
    print("   🔒 Regularización L2 suave: 0.005 → 0.005 → 0.003")
    print("   💧 Dropout moderado: 0.2 → 0.3 → 0.4 → 0.2")
    print("   🧠 Capas dense ampliadas: 768 → 384 → 192")

    return model, base_model


def compile_model(model: Model, learning_rate=LEARNING_RATE) -> Model:
    """Compile the model with appropriate optimizer, loss function and metrics."""

    # For VGG16: use adaptive anti-overfitting learning rate
    if MODEL_ARCHITECTURE.lower() == "vgg16":
        sample_size_typed = auto_type(SAMPLE_SIZE)

        # Detect if large dataset to apply ultra-conservative learning rate
        if (
            sample_size_typed == "full"
            or sample_size_typed == "half"
            or (
                isinstance(sample_size_typed, (int, float))
                and (
                    (
                        isinstance(sample_size_typed, float)
                        and 0 < sample_size_typed <= 1
                        and int(sample_size_typed * 59807) >= 10000
                    )
                    or (
                        isinstance(sample_size_typed, int)
                        and sample_size_typed >= 10000
                    )
                )
            )
        ):
            # Ultra-conservative learning rate for large dataset (anti-overfitting)
            learning_rate = 1e-5  # Extremely conservative vs 1e-4 standard
            print(
                "🛡️  VGG16 Anti-Overfitting: LR ultra-conservador",
                f"{learning_rate} para dataset grande",
            )
        else:
            # Moderate learning rate for small/medium dataset (successful configuration)
            learning_rate = min(
                learning_rate, 0.0001
            )  # 1e-4 - configuration that worked well
            print(
                "🔧 VGG16 configuración exitosa: LR conservador",
                f"{learning_rate} para dataset pequeño/mediano",
            )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,  # Conservative momentum
            beta_2=0.999,
            epsilon=1e-07,  # Numerical stability
            clipnorm=1.0,  # Gradient clipping to prevent explosion
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            DiseaseRecallMetric(),  # Custom metric for disease recall
            keras.metrics.AUC(name="auc"),
        ],
    )

    print("✅ Modelo compilado")
    if MODEL_ARCHITECTURE.lower() == "efficientnet":
        print(
            "🔧 EfficientNet V2.0: Gradient clipping balanceado (clipnorm=0.7, clipvalue=0.7)"
        )
    print("\n📋 Resumen del modelo:")
    model.summary()
    return model


def train_model(
    model: Model,
    train_dataset,
    train_labels,
    val_dataset,
    val_labels,
    class_weights: dict,
    fine_tune: bool = True,
) -> Tuple[Model, dict]:
    """Train the model with two-phase approach.

    Phase 1: Training with frozen base model (15 epochs)

    Args:
        model: Keras model to train.
        train_dataset: Training dataset.
        train_labels: Training labels.
        val_dataset: Validation dataset.
        val_labels: Validation labels for custom metrics.
        class_weights: Class weights to handle imbalance.
        fine_tune: Whether to perform fine-tuning in phase 2.

    Returns:
        Tuple of trained model and training history.
    """
    checkpoint_filename = f"{LOCAL_REGISTRY_PATH}/checkpoints/best_model_{MODEL_ARCHITECTURE}_{SAMPLE_NAME}.keras"

    class RecallFocusedCallback(keras.callbacks.Callback):
        """Custom callback to monitor and report disease recall every 3 epochs."""

        def __init__(self, validation_data, class_names, val_labels):
            super().__init__()
            self.validation_data = validation_data
            self.class_names = class_names
            self.val_labels = val_labels
            self.best_disease_recall = 0.0

        def on_epoch_end(self, epoch, logs=None):
            """Calculate and display disease recall at end of every 3rd epoch."""
            if (epoch + 1) % 3 == 0:  # Every 3 epochs
                # Predict on validation
                y_pred = self.model.predict(self.validation_data, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)

                # Calculate recall for DISEASES (excluding healthy)
                disease_recalls = []
                print(f"\n{'='*50}")
                print(f"📊 RECALL DETALLADO - Epoch {epoch + 1}")
                print(f"{'='*50}")

                for idx, class_name in enumerate(self.class_names):
                    mask = self.val_labels == idx
                    if np.sum(mask) > 0:
                        recall = np.sum(y_pred_classes[mask] == idx) / np.sum(mask)

                        if class_name != "healthy":
                            disease_recalls.append(recall)

                        emoji = "🌱" if class_name == "healthy" else "🦠"
                        print(f"  {emoji} {class_name:15s}: {recall:.4f}")

                # Display how many disease samples were correctly classified by disease
                print("\n✅ Muestras de enfermedad clasificadas correctamente:")
                for idx, class_name in enumerate(self.class_names):
                    if class_name != "healthy":
                        mask = self.val_labels == idx
                        total_should_be_tagged = np.sum(
                            mask
                        )  # Total that should be tagged
                        correct_count = np.sum(
                            y_pred_classes[mask] == idx
                        )  # Correctly classified
                        percentage = (
                            (correct_count / total_should_be_tagged) * 100
                            if total_should_be_tagged > 0
                            else 0
                        )
                        print(
                            f"  🦠 {class_name:15s}: {correct_count} / {total_should_be_tagged}",
                            f"- ({percentage:.2f}%)",
                        )

                # Average disease recall
                avg_disease_recall = np.mean(disease_recalls) if disease_recalls else 0
                print(f"\n🎯 Recall promedio enfermedades: {avg_disease_recall:.4f}")

                if avg_disease_recall > self.best_disease_recall:
                    self.best_disease_recall = avg_disease_recall
                    print("✨ ¡Nuevo mejor recall de enfermedades!")

    # Configure architecture-specific callbacks
    if MODEL_ARCHITECTURE.lower() == "vgg16":
        # VGG16: more conservative callbacks to avoid oscillations
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",  # For VGG16, monitoring loss is more stable
                patience=8,  # Less patience to avoid overfitting
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",  # Use loss for greater stability
                factor=0.5,  # More gradual reduction
                patience=3,  # Reduce LR faster
                min_lr=1e-7,
                mode="min",
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_filename,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            RecallFocusedCallback(val_dataset, CLASS_NAMES, val_labels),
        ]
    else:
        # Original callbacks for CNN and EfficientNet
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_recall",  # Keep recall as primary monitor
                patience=15,  # Use more patience for small datasets
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_recall",
                factor=0.3,
                patience=5,
                min_lr=1e-8,
                mode="max",
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_filename,
                monitor="val_recall",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            RecallFocusedCallback(val_dataset, CLASS_NAMES, val_labels),
        ]

    print("\n" + "=" * 60)
    if MODEL_ARCHITECTURE.lower() == "vgg16":
        print("🚂 ENTRENAMIENTO VGG16: Transfer Learning conservador (20 epochs)")
        print("ℹ️  Estrategia: Entrenamiento gradual con learning rate bajo")
    elif MODEL_ARCHITECTURE.lower() != "cnn":
        print(f"🚂 FASE 1: Entrenando con {MODEL_ARCHITECTURE} congelado (15 epochs)")
    else:
        print("🚂 ENTRENAMIENTO: Modelo CNN simple (30 epochs)")
    print("=" * 60)

    # Adjust epochs by model type and dataset size
    if MODEL_ARCHITECTURE.lower() == "vgg16":
        # VGG16: fewer epochs to avoid overfitting on 6K images
        initial_epochs = 20
    elif MODEL_ARCHITECTURE.lower() == "cnn":
        initial_epochs = 30
    else:
        initial_epochs = 15

    history_phase1 = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("✅ Fase 1 de entrenamiento completada")
    print(
        "📈 Recall máximo en validación durante fase 1:",
        f"{max(history_phase1.history['val_recall']):.4f}",
    )

    # For VGG16 with 6K images: NO fine-tuning to avoid overfitting
    should_finetune = (
        MODEL_ARCHITECTURE.lower() == "efficientnet"  # Only EfficientNet
        and len(train_labels) >= 10000  # Large dataset
        and len(history_phase1.history["val_accuracy"]) > 0
        and max(history_phase1.history["val_accuracy"])
        > 0.60  # Minimum accuracy threshold
        and fine_tune
    )

    if should_finetune:
        model, history_phase2 = fine_tune_model(
            model,
            model.layers[1],  # base_model está en la segunda posición
            history_phase1,
            train_dataset,
            val_dataset,
            class_weights,
            callbacks,
        )
        combined_history = combine_histories(history_phase1, history_phase2)
    else:
        if MODEL_ARCHITECTURE.lower() == "vgg16":
            print("\n" + "=" * 60)
            print("ℹ️  VGG16: No fine-tuning para dataset de 6K imágenes")
            print("ℹ️  Transfer learning con capas finales entrenables es suficiente")
            print("=" * 60)
        elif MODEL_ARCHITECTURE != "cnn":
            print("\n" + "=" * 60)
            print(
                f"⚠️  FASE 2: Saltando fine-tuning (dataset pequeño: {len(train_labels)} imágenes)"
            )
            print("=" * 60)
            print("ℹ️  Se requieren al menos 10,000 imágenes para fine-tuning seguro.")
            print(
                "ℹ️  El modelo se mantiene con la base congelada (solo la cabeza entrenada)."
            )
        else:
            print("\n" + "=" * 60)
            print("ℹ️  Modelo CNN simple: No requiere fine-tuning")
            print("=" * 60)

        # Create empty history_phase2 to avoid errors
        combined_history = history_phase1

    return model, combined_history


def fine_tune_model(
    model: Model,
    base_model,
    history_phase1,
    train_dataset,
    val_dataset,
    class_weights: dict,
    callbacks,
):
    """Perform fine-tuning by unfreezing the last layers of EfficientNet.

    Args:
        model: Keras model to train.
        base_model: The EfficientNet base of the model.
        history_phase1: Previous training history.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        class_weights: Class weights to handle imbalance.
        callbacks: Callbacks for training.

    Returns:
        Tuple of trained model and training history.
    """
    print("\n" + "=" * 60)
    if MODEL_ARCHITECTURE.lower() == "efficientnet":
        print("🔥 FASE 2: Fine-tuning BALANCEADO (EfficientNet V2.0)")
        print("🎯 8 capas finales para mejor performance manteniendo estabilidad")
    else:
        print("🔥 FASE 2: Fine-tuning (descongelando últimas 15 capas)")
    print("=" * 60)
    print(
        f"ℹ️  Mejor val_accuracy en Fase 1: {max(history_phase1.history['val_accuracy']):.3f}"
    )

    # Verify stability before fine-tuning for EfficientNet
    if MODEL_ARCHITECTURE.lower() == "efficientnet":
        val_loss_history = history_phase1.history["val_loss"]
        best_val_loss = min(val_loss_history)
        recent_val_loss = val_loss_history[-3:]  # Last 3 epochs

        print("📊 Análisis de estabilidad V2.0:")
        print(f"   Mejor val_loss: {best_val_loss:.4f}")
        print(f"   Val_loss reciente: {recent_val_loss}")

        # More permissive conditions for fine-tuning
        if best_val_loss > 0.4 or any(
            loss > best_val_loss * 1.8 for loss in recent_val_loss
        ):
            print("⚠️  Modelo muestra inestabilidad - usando fine-tuning conservador")
            conservative_mode = True
        else:
            print("✅ Modelo estable - usando fine-tuning balanceado")
            conservative_mode = False
    else:
        conservative_mode = False

    # Unfreeze base model
    base_model.trainable = True

    # Balanced fine-tuning for EfficientNet V2.0
    if MODEL_ARCHITECTURE.lower() == "efficientnet":
        if conservative_mode:
            fine_tune_at = len(base_model.layers) - 5  # Only 5 layers if unstable
            lr_divisor = 15  # Learning rate 15x lower
            print("🚨 Modo conservador: solo 5 capas finales")
        else:
            fine_tune_at = len(base_model.layers) - 8  # 8 layers (vs 5 previous)
            lr_divisor = 10  # Learning rate 10x lower (vs 15-20)
            print("🔧 Modo balanceado: 8 capas finales")
    else:
        fine_tune_at = len(base_model.layers) - 15  # Original value for other models
        lr_divisor = 10

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable_layers = sum([1 for l in model.layers if l.trainable])
    frozen_layers = len(model.layers) - trainable_layers

    print(f"🔒 Capas congeladas: {frozen_layers}")
    print(f"🔥 Capas entrenables: {trainable_layers}")
    print(f"📊 Total capas: {len(model.layers)}")

    # Architecture-specific learning rate
    new_lr = LEARNING_RATE / lr_divisor
    print(f"📉 Learning rate reducido: {LEARNING_RATE} → {new_lr} (÷{lr_divisor})")

    # Recompile with optimized configurations by architecture
    if MODEL_ARCHITECTURE.lower() == "efficientnet":
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=new_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                clipnorm=0.5,  # Balanced gradient clipping (0.5 vs 0.3)
                clipvalue=0.5,  # Also clip by balanced value
            ),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Recall(name="recall"),
                keras.metrics.Precision(name="precision"),
                DiseaseRecallMetric(),
                keras.metrics.AUC(name="auc"),
            ],
        )
        print(
            "🔧 EfficientNet V2.0: Gradient clipping balanceado (0.5) para fine-tuning"
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=new_lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Recall(name="recall"),  # General recall
                keras.metrics.Precision(name="precision"),
                DiseaseRecallMetric(),  # Disease-specific recall
                keras.metrics.AUC(name="auc"),
            ],
        )

    history_phase2 = model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=len(history_phase1.epoch),
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history_phase2


def evaluate_model(
    model: Model,
    test_dataset,
    test_labels,
    get_confusion_matrix: bool = False,
    get_false_negatives_analysis: bool = True,
) -> Tuple[Model, dict]:
    """Evaluate the model on a given dataset.

    Args:
        model: Keras model to evaluate.
        test_dataset: Test dataset.
        test_labels: True labels from test set.
        get_confusion_matrix: Whether to generate confusion matrix.
        get_false_negatives_analysis: Whether to perform false negatives analysis.

    Returns:
        Tuple of evaluation results.
    """

    if model is None:
        print(
            Fore.RED
            + "❌ Modelo no está definido. No se puede evaluar."
            + Style.RESET_ALL
        )
        return None

    model_name = MODEL_NAME

    print("\n" + "=" * 60)
    print("🧪 EVALUACIÓN FINAL EN TEST SET")
    print("=" * 60)

    # Evaluar en test
    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

    # Predecir en test
    y_pred_test = model.predict(test_dataset, verbose=1)
    y_pred_test_classes = np.argmax(y_pred_test, axis=1)

    # Identify classes present in test
    unique_test_classes = np.unique(test_labels)
    print(f"\nClases presentes en test set: {unique_test_classes}")

    # Check if all classes are present
    missing_classes = set(range(NUM_CLASSES)) - set(unique_test_classes)
    if missing_classes:
        print(
            f"⚠️  Clases ausentes en test: {[CLASS_NAMES[i] for i in missing_classes]}"
        )

    # Classification report
    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 60)

    # Option: Specify only present classes (recommended for small samples)
    if len(unique_test_classes) < NUM_CLASSES:
        # Use only names of present classes
        target_names_present = [CLASS_NAMES[i] for i in unique_test_classes]
        print(
            classification_report(
                test_labels,
                y_pred_test_classes,
                labels=unique_test_classes,
                target_names=target_names_present,
                digits=4,
            )
        )
        print(
            f"\n⚠️  Nota: Solo se muestran las {len(unique_test_classes)}",
            "clases presentes en el test set.",
        )
    else:
        # All classes present, use complete report
        print(
            classification_report(
                test_labels, y_pred_test_classes, target_names=CLASS_NAMES, digits=4
            )
        )

    if get_confusion_matrix:
        # Confusion matrix
        cm = confusion_matrix(
            test_labels, y_pred_test_classes, labels=unique_test_classes
        )

        # Use only names of present classes for axes
        axis_labels = [CLASS_NAMES[i] for i in unique_test_classes]

        # Descriptive name for confusion matrix
        confusion_matrix_filename = f"{MODELS_PATH}/confusion_matrix_{model_name}.png"

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
            f"Matriz de Confusión - Test Set\n{model_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Etiqueta Real", fontsize=12)
        plt.xlabel("Predicción", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(confusion_matrix_filename, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"💾 Matriz de confusión guardada: {confusion_matrix_filename}")

    if get_false_negatives_analysis:
        # Detailed false negatives analysis
        false_negatives_analysis(test_labels, y_pred_test_classes)

    return test_results
