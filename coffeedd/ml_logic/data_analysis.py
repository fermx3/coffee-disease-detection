import os
from colorama import Fore, Style
import numpy as np
import matplotlib.pyplot as plt

from coffeedd.params import CLASS_NAMES, EPOCHS, MODELS_PATH, NUM_CLASSES


def find_rarest_disease_class(data_path, class_names, extreme_threshold=0.5):
    """Identify the rarest disease class when extreme imbalance exists.

    This inspects the filesystem under ``data_path`` for each class folder in
    ``class_names`` (ignoring 'healthy') and computes simple statistics.

    Args:
        data_path: Path to the dataset directory.
        class_names: Iterable of class names (strings).
        extreme_threshold: Float threshold used to decide extreme rarity
            relative to the average (e.g. 0.5 means "less than half the average").

    Returns:
        tuple: (rarest_class_name or None, min_count, is_extreme_bool, stats_dict)
    """

    # Identify disease classes (exclude healthy)
    disease_classes = [name for name in CLASS_NAMES if name != "healthy"]
    num_disease_classes = len(disease_classes)
    class_counts = {}

    for class_name in class_names:
        if class_name == "healthy":
            continue

        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            count = len(
                [
                    f
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            class_counts[class_name] = count

    if not class_counts:
        return None, 0, False, {}

    # Compute statistics
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)

    # Rarest class name
    rarest_class = min(class_counts, key=class_counts.get)

    # Compute ratios for heuristic extreme-imbalance detection
    ratio_vs_avg = min_count / avg_count if avg_count > 0 else 1
    ratio_vs_max = min_count / max_count if max_count > 0 else 1

    # Extreme if much smaller than the average AND much smaller than the max
    is_extreme = ratio_vs_avg < extreme_threshold and ratio_vs_max < 0.3

    stats = {
        "all_counts": class_counts,
        "min_count": min_count,
        "max_count": max_count,
        "avg_count": avg_count,
        "ratio_vs_avg": ratio_vs_avg,
        "ratio_vs_max": ratio_vs_max,
        "is_extreme": is_extreme,
    }

    # Print an informational header (Spanish intentionally preserved)
    print("\n🎯 Configuración optimizada para DETECCIÓN DE ENFERMEDADES:")
    print("  📊 Objetivo: Minimizar falsos negativos (enfermedad → healthy)")
    print(
        f"  🦠 Clases de enfermedad detectadas: {num_disease_classes} ({', '.join(disease_classes)})"
    )
    print("  🌱 Clase healthy: Será penalizada para evitar falsos negativos")
    print("  📈 Métrica principal: Recall")
    print(f"  ⏰ Epochs configurados: {EPOCHS}")
    print("\n⏳ Class weights se calcularán después de cargar los datos...")

    return rarest_class, min_count, is_extreme, stats


def false_negatives_analysis(test_labels, y_pred_test_classes):
    """Detailed false-negative analysis for the test set.

    Args:
        test_labels: Array-like of true labels for the test set.
        y_pred_test_classes: Array-like of predicted class indices for the test set.
    """
    print("\n" + "=" * 60)
    print("⚠️  ANÁLISIS DETALLADO DE FALSOS NEGATIVOS")
    print("=" * 60)

    healthy_idx = CLASS_NAMES.index("healthy")

    print("\n⚠️  Falsos Negativos (Enfermedad → Healthy):")

    total_fn = 0
    total_disease_samples = 0

    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name == "healthy":
            continue  # Saltar la clase healthy
        # Mask for true cases of this disease
        mask_true = test_labels == idx
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            # Mask for model predictions equal to 'healthy'
            mask_pred_healthy = y_pred_test_classes == healthy_idx

            # False negatives: true disease predicted as healthy
            fn = np.sum(mask_true & mask_pred_healthy)
            fn_rate = (fn / total_cases) * 100

            print(f"  {class_name:20s}: {fn}/{total_cases} ({fn_rate:.1f}%)")

            total_fn += fn
            total_disease_samples += total_cases

    print(f"\n🔴 Total Falsos Negativos: {total_fn}")
    print(f"📊 Total casos de enfermedad en test: {total_disease_samples}")

    if total_disease_samples > 0:
        overall_fn_rate = (total_fn / total_disease_samples) * 100
        print(f"📈 Tasa global de Falsos Negativos: {overall_fn_rate:.1f}%")

    # Additional analysis: which classes do diseases get confused with?
    print("\n🔍 Análisis de confusiones por enfermedad:")
    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name == "healthy":
            continue

        mask_true = test_labels == idx
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            print(f"\n  {class_name} (total: {total_cases}):")
            predictions_for_this_class = y_pred_test_classes[mask_true]

            for pred_idx, pred_class in enumerate(CLASS_NAMES):
                count = np.sum(predictions_for_this_class == pred_idx)
                if count > 0:
                    percentage = (count / total_cases) * 100
                    emoji = (
                        "✅"
                        if pred_idx == idx
                        else ("❌" if pred_class == "healthy" else "🔄")
                    )
                    print(
                        f"    {emoji} → {pred_class:15s}: {count}/{total_cases} ({percentage:.1f}%)"
                    )


def analyze_false_negatives(test_labels, y_pred_test_classes, verbose=True):
    """Analyze false negatives and build MLflow-friendly metrics.

    Focuses on disease -> healthy false negatives and per-class rates.

    Args:
        test_labels: Array-like of true labels for the test set.
        y_pred_test_classes: Array-like of predicted class indices for the test set.
        verbose: If True, print a human-readable report.

    Returns:
        dict: Aggregated metrics suitable for logging to MLflow.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(Fore.RED + "⚠️  ANÁLISIS DETALLADO DE FALSOS NEGATIVOS" + Style.RESET_ALL)
        print("=" * 60)

    # Find index for the 'healthy' class
    healthy_idx = None
    for idx, class_name in enumerate(CLASS_NAMES):
        if "healthy" in class_name.lower() or "sano" in class_name.lower():
            healthy_idx = idx
            break

    if healthy_idx is None:
        if verbose:
            print(
                f"{Fore.YELLOW}⚠️  No se encontró clase 'healthy' en {CLASS_NAMES}{Style.RESET_ALL}"
            )
        # Fallback: assume first class is healthy
        healthy_idx = 0
        if verbose:
            print(
                f"{Fore.YELLOW}📝 Asumiendo que '{CLASS_NAMES[0]}' es la clase healthy{Style.RESET_ALL}"
            )

    if verbose:
        print(
            f"\n{Fore.RED}⚠️  Falsos Negativos (Enfermedad → {CLASS_NAMES[healthy_idx]}):{Style.RESET_ALL}"
        )

    # Métricas para retornar
    fn_metrics = {}
    total_fn = 0
    total_disease_samples = 0
    class_fn_rates = {}

    # Analyze each disease class
    for idx, class_name in enumerate(CLASS_NAMES):
        if idx == healthy_idx:
            continue  # Saltar la clase healthy

        # Máscara para casos reales de esta enfermedad
        mask_true = test_labels == idx
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            # Mask for predictions equal to 'healthy'
            mask_pred_healthy = y_pred_test_classes == healthy_idx

            # False negatives: true disease predicted as healthy
            fn = np.sum(mask_true & mask_pred_healthy)
            fn_rate = (fn / total_cases) * 100

            if verbose:
                print(f"  {class_name:20s}: {fn}/{total_cases} ({fn_rate:.1f}%)")

            # Guardar métricas
            fn_metrics[f"fn_{class_name}"] = int(fn)
            fn_metrics[f"fn_rate_{class_name}"] = fn_rate
            fn_metrics[f"total_cases_{class_name}"] = int(total_cases)
            class_fn_rates[class_name] = fn_rate

            total_fn += fn
            total_disease_samples += total_cases

            if verbose:
                print(f"\n{Fore.RED}🔴 Total Falsos Negativos: {total_fn}{Style.RESET_ALL}")
                print(
                    f"{Fore.BLUE}📊 Total casos de enfermedad en test: {total_disease_samples}{Style.RESET_ALL}"
                )

    # Métricas globales
    overall_fn_rate = 0
    if total_disease_samples > 0:
        overall_fn_rate = (total_fn / total_disease_samples) * 100
        if verbose:
            print(
                f"{Fore.MAGENTA}📈 Tasa global de Falsos Negativos: {overall_fn_rate:.1f}%{Style.RESET_ALL}"
            )

    # Análisis adicional: ¿A qué clases se confunden las enfermedades?
    if verbose:
        print(
            f"\n{Fore.CYAN}🔍 Análisis de confusiones por enfermedad:{Style.RESET_ALL}"
        )

    confusion_details = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        if idx == healthy_idx:
            continue

        mask_true = test_labels == idx
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            if verbose:
                print(
                    f"\n  {Fore.YELLOW}{class_name}{Style.RESET_ALL} (total: {total_cases}):"
                )

            predictions_for_this_class = y_pred_test_classes[mask_true]
            class_confusions = {}

            for pred_idx, pred_class in enumerate(CLASS_NAMES):
                count = np.sum(predictions_for_this_class == pred_idx)
                if count > 0:
                    percentage = (count / total_cases) * 100

                    # Guardar en métricas
                    confusion_key = f"confusion_{class_name}_to_{pred_class}"
                    confusion_details[confusion_key] = percentage
                    class_confusions[pred_class] = percentage

                    if verbose:
                        if pred_idx == idx:
                            emoji = "✅"
                            color = Fore.GREEN
                        elif pred_class.lower() == CLASS_NAMES[healthy_idx].lower():
                            emoji = "❌"
                            color = Fore.RED
                        else:
                            emoji = "🔄"
                            color = Fore.YELLOW

                        print(
                            f"    {emoji} → {color}{pred_class:15s}{Style.RESET_ALL}: {count}/{total_cases} ({percentage:.1f}%)"
                        )

    # Compilar todas las métricas para MLflow
    mlflow_metrics = {
        # Métricas globales de falsos negativos
        "total_false_negatives": total_fn,
        "total_disease_samples": total_disease_samples,
        "overall_false_negative_rate": overall_fn_rate,
        # Métricas por clase
        **fn_metrics,
        # Detalles de confusión (solo las más importantes para no saturar MLflow)
        **{
            k: v for k, v in confusion_details.items() if v > 5.0
        },  # Solo confusiones > 5%
        # Estadísticas adicionales
        "classes_with_fn": len([rate for rate in class_fn_rates.values() if rate > 0]),
        "max_fn_rate": max(class_fn_rates.values()) if class_fn_rates else 0,
        "min_fn_rate": min(class_fn_rates.values()) if class_fn_rates else 0,
        "avg_fn_rate": np.mean(list(class_fn_rates.values())) if class_fn_rates else 0,
    }

    if verbose:
        print("\n" + "=" * 60)
        print(
            Fore.GREEN + "✅ ANÁLISIS DE FALSOS NEGATIVOS COMPLETADO" + Style.RESET_ALL
        )
        print("=" * 60)
        print(f"{Fore.GREEN}📋 Resumen para MLflow:{Style.RESET_ALL}")
        print(f"   • Métricas registradas: {len(mlflow_metrics)}")
        print(f"   • Tasa FN global: {overall_fn_rate:.1f}%")
        print(
            f"   • Clases con FN: {len([rate for rate in class_fn_rates.values() if rate > 0])}"
        )

    return mlflow_metrics


def analyze_disease_recall(test_labels, y_pred_test_classes, verbose=True):
    """Compute disease-detection metrics treating all diseases vs healthy.

    Converts multi-class predictions to binary (disease vs healthy) and
    returns recall/precision/F1/accuracy plus counts.
    """
    # Encontrar el índice de la clase 'healthy'
    healthy_idx = None
    for idx, class_name in enumerate(CLASS_NAMES):
        if "healthy" in class_name.lower() or "sano" in class_name.lower():
            healthy_idx = idx
            break

    if healthy_idx is None:
        healthy_idx = 0  # Asumir que la primera clase es healthy

    # Convertir a binary: 0 = healthy, 1 = cualquier enfermedad
    binary_true = (test_labels != healthy_idx).astype(int)
    binary_pred = (y_pred_test_classes != healthy_idx).astype(int)

    # Calcular métricas binarias
    true_positives = np.sum((binary_true == 1) & (binary_pred == 1))
    false_negatives = np.sum((binary_true == 1) & (binary_pred == 0))
    false_positives = np.sum((binary_true == 0) & (binary_pred == 1))
    true_negatives = np.sum((binary_true == 0) & (binary_pred == 0))

    # Calcular métricas
    disease_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    disease_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    disease_f1 = (
        2 * (disease_precision * disease_recall) / (disease_precision + disease_recall)
        if (disease_precision + disease_recall) > 0
        else 0
    )
    disease_accuracy = (true_positives + true_negatives) / len(test_labels)

    if verbose:
        print(
            f"\n{Fore.CYAN}🩺 MÉTRICAS DE DETECCIÓN DE ENFERMEDADES (Binary Classification){Style.RESET_ALL}"
        )
        print("=" * 60)
        print(f"   • Disease Recall (Sensibilidad): {disease_recall:.4f}")
        print(f"   • Disease Precision: {disease_precision:.4f}")
        print(f"   • Disease F1-Score: {disease_f1:.4f}")
        print(f"   • Disease Accuracy: {disease_accuracy:.4f}")
        print(f"\n   • True Positives (Enfermedad detectada): {true_positives}")
        print(f"   • False Negatives (Enfermedad perdida): {false_negatives}")
        print(f"   • False Positives (Falsa alarma): {false_positives}")
        print(f"   • True Negatives (Healthy correcto): {true_negatives}")

    return {
        "disease_recall": disease_recall,
        "disease_precision": disease_precision,
        "disease_f1_score": disease_f1,
        "disease_accuracy": disease_accuracy,
        "disease_true_positives": true_positives,
        "disease_false_negatives": false_negatives,
        "disease_false_positives": false_positives,
        "disease_true_negatives": true_negatives,
        "total_disease_cases": int(np.sum(binary_true)),
        "total_healthy_cases": int(np.sum(1 - binary_true)),
    }


def detect_fine_tuning_start(combined_history, verbose=True):
    """Detect whether training includes a fine-tuning phase.

    Heuristics look for sudden drops in loss (train or validation) and
    prefer detections within a plausible epoch range.
    """
    loss_values = combined_history.history.get("loss", [])
    val_loss_values = combined_history.history.get("val_loss", [])

    if len(loss_values) < 10:  # Too few epochs to detect fine-tuning
        return False, None

    # Detect sudden drops in loss (typical when layers are unfrozen)
    loss_diffs = np.diff(loss_values)

    # Buscar caídas significativas en la pérdida (typical cuando se descongelan capas)
    significant_drops = []
    for i, diff in enumerate(loss_diffs[5:], start=5):  # Empezar después de epoch 5
        # Detectar caída súbita > 20% del valor actual
        if diff < -0.2 * loss_values[i] and abs(diff) > 0.1:
            significant_drops.append(i + 1)  # +1 porque diff está desplazado

    # Also examine validation loss
    if val_loss_values:
        val_loss_diffs = np.diff(val_loss_values)
        for i, diff in enumerate(val_loss_diffs[5:], start=5):
            if diff < -0.2 * val_loss_values[i] and abs(diff) > 0.1:
                significant_drops.append(i + 1)

    # Look for likely fine-tuning epochs in a plausible range
    likely_fine_tune_epochs = [e for e in significant_drops if 10 <= e <= 25]

    if likely_fine_tune_epochs:
        fine_tune_start = min(likely_fine_tune_epochs)
        if verbose:
            print(
                f"🔍 Fine-tuning detectado automáticamente en epoch {fine_tune_start}"
            )
        return True, fine_tune_start

    # Fallback: if training is long, assume fine-tuning starts at epoch 15
    if len(loss_values) > 20:
        if verbose:
            print(
                "🔍 Asumiendo fine-tuning en epoch 15 (entrenamiento largo detectado)"
            )
        return True, 15

    return False, None


def plot_training_metrics_combined(
    combined_history,
    model_name,
    sample_name,
    test_labels=None,
    y_pred_test_classes=None,
    verbose=True,
):
    """Generate comprehensive training metric visualizations from history.

    This function builds accuracy/loss/recall plots plus a per-class recall
    bar chart (if test predictions are provided). It saves a PNG to
    ``MODELS_PATH`` and returns a metrics dict suitable for MLflow logging.

    Notes:
    - Human-facing prints are intentionally left in Spanish to match the
      project's CLI style.

    Args:
        combined_history: Keras History-like object containing training metrics.
        model_name: String used for plot titles and saved filename.
        sample_name: Sample name inserted into plot titles.
        test_labels: Optional array-like of true test labels.
        y_pred_test_classes: Optional array-like of predicted test classes.
        verbose: If True, print progress and summary information.

    Returns:
        dict of aggregated metrics for MLflow.
    """
    if verbose:
        print("\n" + "=" * 60)
        print(
            Fore.CYAN
            + "📊 GENERANDO VISUALIZACIONES DE ENTRENAMIENTO"
            + Style.RESET_ALL
        )
        print("=" * 60)

    # Detectar automáticamente si hubo fine-tuning
    has_fine_tuning, fine_tune_start_epoch = detect_fine_tuning_start(
        combined_history, verbose
    )

    # Extraer métricas del historial combinado
    accuracy = combined_history.history.get("accuracy", [])
    val_accuracy = combined_history.history.get("val_accuracy", [])
    loss = combined_history.history.get("loss", [])
    val_loss = combined_history.history.get("val_loss", [])
    recall = combined_history.history.get("recall", [])
    val_recall = combined_history.history.get("val_recall", [])

    # Verificar que tenemos datos
    if not accuracy:
        if verbose:
            print(
                f"{Fore.RED}❌ Error: No hay datos de entrenamiento para graficar{Style.RESET_ALL}"
            )
        return {}
    else:
        if verbose:
            print(
                f"{Fore.GREEN}✅ Datos disponibles: {len(accuracy)} epochs{Style.RESET_ALL}"
            )
            if has_fine_tuning:
                print(
                    f"{Fore.YELLOW}🔧 Fine-tuning detectado desde epoch {fine_tune_start_epoch}{Style.RESET_ALL}"
                )

    # Crear directorio de modelos si no existe
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Nombre descriptivo para las métricas
    metrics_filename = f"{MODELS_PATH}/training_metrics_{model_name}.png"

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ==========================================
    # 1. ACCURACY PLOT
    # ==========================================
    if accuracy and val_accuracy:
        epochs_range = range(len(accuracy))
        axes[0, 0].plot(
            epochs_range, accuracy, label="Train", linewidth=2, color="#2E86AB"
        )
        axes[0, 0].plot(
            epochs_range, val_accuracy, label="Validation", linewidth=2, color="#A23B72"
        )

        # Línea de fine-tuning si fue detectado
        if (
            has_fine_tuning
            and fine_tune_start_epoch is not None
            and fine_tune_start_epoch < len(accuracy)
        ):
            axes[0, 0].axvline(
                x=fine_tune_start_epoch,
                color="red",
                linestyle="--",
                label="Fine-tuning start",
                alpha=0.7,
                linewidth=2,
            )

        axes[0, 0].set_title(f"Accuracy - {model_name}", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No accuracy data available",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
            fontsize=14,
        )
        axes[0, 0].set_title("Accuracy - No Data")

    # ==========================================
    # 2. LOSS PLOT
    # ==========================================
    if loss and val_loss:
        epochs_range = range(len(loss))
        axes[0, 1].plot(epochs_range, loss, label="Train", linewidth=2, color="#2E86AB")
        axes[0, 1].plot(
            epochs_range, val_loss, label="Validation", linewidth=2, color="#A23B72"
        )

        if (
            has_fine_tuning
            and fine_tune_start_epoch is not None
            and fine_tune_start_epoch < len(loss)
        ):
            axes[0, 1].axvline(
                x=fine_tune_start_epoch,
                color="red",
                linestyle="--",
                label="Fine-tuning start",
                alpha=0.7,
                linewidth=2,
            )

        axes[0, 1].set_title(f"Loss - {model_name}", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No loss data available",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=14,
        )
        axes[0, 1].set_title("Loss - No Data")

    # ==========================================
    # 3. RECALL PLOT
    # ==========================================
    if recall and val_recall:
        epochs_range = range(len(recall))
        axes[1, 0].plot(
            epochs_range, recall, label="Train", linewidth=2, color="#2E86AB"
        )
        axes[1, 0].plot(
            epochs_range, val_recall, label="Validation", linewidth=2, color="#A23B72"
        )

        if (
            has_fine_tuning
            and fine_tune_start_epoch is not None
            and fine_tune_start_epoch < len(recall)
        ):
            axes[1, 0].axvline(
                x=fine_tune_start_epoch,
                color="red",
                linestyle="--",
                label="Fine-tuning start",
                alpha=0.7,
                linewidth=2,
            )

        axes[1, 0].set_title(f"Recall - {model_name}", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No recall data available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=14,
        )
        axes[1, 0].set_title("Recall - No Data")

    # ==========================================
    # 4. RECALL POR CLASE (Test Set)
    # ==========================================
    if test_labels is not None and y_pred_test_classes is not None:
        recall_per_class = []
        class_names_present = []

        for idx in range(NUM_CLASSES):
            mask = test_labels == idx
            if np.sum(mask) > 0:  # Solo incluir clases que existen en test
                class_recall = np.sum(y_pred_test_classes[mask] == idx) / np.sum(mask)
                recall_per_class.append(class_recall)
                class_names_present.append(CLASS_NAMES[idx])

        if recall_per_class:
            # Colores más bonitos
            colors = [
                "#27AE60" if cn.lower() in ["healthy", "sano"] else "#E74C3C"
                for cn in class_names_present
            ]
            bars = axes[1, 1].bar(
                range(len(recall_per_class)),
                recall_per_class,
                color=colors,
                edgecolor="black",
                alpha=0.8,
            )
            axes[1, 1].set_xticks(range(len(class_names_present)))
            axes[1, 1].set_xticklabels(class_names_present, rotation=45, ha="right")
            axes[1, 1].set_title(
                f"Recall por Clase - Test Set\n{sample_name}",
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].grid(True, axis="y", alpha=0.3)

            # Agregar valores en las barras
            for bar, val in zip(bars, recall_per_class):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No test data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
            )
            axes[1, 1].set_title("Recall por Clase - No Data")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Test evaluation not completed",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=14,
        )
        axes[1, 1].set_title("Recall por Clase - Pending")

    # Use the Figure object directly to layout and save (avoids unused var)
    fig.tight_layout()

    # Save the figure
    try:
        fig.savefig(metrics_filename, dpi=300, bbox_inches="tight")
        if verbose:
            print(
                f"{Fore.GREEN}💾 Métricas de entrenamiento guardadas: {metrics_filename}{Style.RESET_ALL}"
            )
        plt.close(fig)  # Close the specific figure to free memory
    except Exception as e:
        if verbose:
            print(f"{Fore.RED}⚠️ Error al guardar gráfico: {e}{Style.RESET_ALL}")
        plt.close(fig)

    # ==========================================
    # CALCULAR MÉTRICAS PARA MLFLOW
    # ==========================================

    mlflow_metrics = {}

    # Métricas de entrenamiento (último epoch)
    if accuracy:
        mlflow_metrics.update(
            {
                "final_train_accuracy": accuracy[-1],
                "final_val_accuracy": val_accuracy[-1] if val_accuracy else 0,
                "max_train_accuracy": max(accuracy),
                "max_val_accuracy": max(val_accuracy) if val_accuracy else 0,
                "total_epochs": len(accuracy),
            }
        )

    if loss:
        mlflow_metrics.update(
            {
                "final_train_loss": loss[-1],
                "final_val_loss": val_loss[-1] if val_loss else 0,
                "min_train_loss": min(loss),
                "min_val_loss": min(val_loss) if val_loss else 0,
            }
        )

    if recall:
        mlflow_metrics.update(
            {
                "final_train_recall": recall[-1],
                "final_val_recall": val_recall[-1] if val_recall else 0,
                "max_train_recall": max(recall),
                "max_val_recall": max(val_recall) if val_recall else 0,
            }
        )

    # Información sobre fine-tuning
    mlflow_metrics.update(
        {
            "has_fine_tuning": has_fine_tuning,
            "fine_tune_start_epoch": (
                fine_tune_start_epoch if fine_tune_start_epoch else 0
            ),
            "pre_fine_tune_epochs": (
                fine_tune_start_epoch if fine_tune_start_epoch else len(accuracy)
            ),
            "post_fine_tune_epochs": (
                len(accuracy) - fine_tune_start_epoch if fine_tune_start_epoch else 0
            ),
        }
    )

    # Métricas de recall por clase (si están disponibles)
    if test_labels is not None and y_pred_test_classes is not None:
        for idx in range(NUM_CLASSES):
            mask = test_labels == idx
            if np.sum(mask) > 0:
                class_recall = np.sum(y_pred_test_classes[mask] == idx) / np.sum(mask)
                mlflow_metrics[f"recall_class_{CLASS_NAMES[idx]}"] = class_recall

        # Estadísticas globales de recall por clase
        class_recalls = [
            v for k, v in mlflow_metrics.items() if k.startswith("recall_class_")
        ]
        if class_recalls:
            mlflow_metrics.update(
                {
                    "avg_class_recall": np.mean(class_recalls),
                    "min_class_recall": min(class_recalls),
                    "max_class_recall": max(class_recalls),
                    "std_class_recall": np.std(class_recalls),
                }
            )

    if verbose:
        print("\n" + "=" * 60)
        print(
            Fore.GREEN
            + "✅ VISUALIZACIONES DE ENTRENAMIENTO COMPLETADAS"
            + Style.RESET_ALL
        )
        print("=" * 60)
        print(f"{Fore.GREEN}📋 Resumen para MLflow:{Style.RESET_ALL}")
        print(f"   • Métricas calculadas: {len(mlflow_metrics)}")
        print(f"   • Total epochs: {mlflow_metrics.get('total_epochs', 0)}")
        print(f"   • Fine-tuning detectado: {'Sí' if has_fine_tuning else 'No'}")
        if has_fine_tuning:
            print(f"   • Inicio fine-tuning: Epoch {fine_tune_start_epoch}")
        print(f"   • Archivo guardado: {metrics_filename}")

    return mlflow_metrics


def analyze_training_convergence_combined(combined_history, verbose=True):
    """Analyze training convergence from a combined Keras History.

    Computes heuristics such as overfitting detection, convergence stability,
    and an estimate of fine-tuning benefit. Returns a dictionary suitable
    for MLflow logging.
    """
    if verbose:
        print(
            f"\n{Fore.CYAN}🔍 ANÁLISIS DE CONVERGENCIA DEL ENTRENAMIENTO{Style.RESET_ALL}"
        )
        print("=" * 60)

    # Extract metrics from the combined history
    val_loss = combined_history.history.get("val_loss", [])
    train_loss = combined_history.history.get("loss", [])

    convergence_metrics = {}

    if val_loss and len(val_loss) > 5:
        # Detectar overfitting (val_loss sube en últimas epochs)
        last_5_val_loss = val_loss[-5:]
        is_overfitting = last_5_val_loss[-1] > last_5_val_loss[0]

        # Detectar early stopping necesario
        best_val_loss_epoch = val_loss.index(min(val_loss))
        epochs_since_best = len(val_loss) - best_val_loss_epoch - 1

        # Detectar convergencia
        val_loss_std_last_10 = (
            np.std(val_loss[-10:]) if len(val_loss) >= 10 else float("inf")
        )
        has_converged = val_loss_std_last_10 < 0.01

        # Detectar si fine-tuning ayudó (comparar antes y después)
        has_fine_tuning, fine_tune_start = detect_fine_tuning_start(
            combined_history, verbose=False
        )
        fine_tuning_improvement = 0

        if has_fine_tuning and fine_tune_start and fine_tune_start < len(val_loss) - 5:
            pre_fine_tune_best = min(val_loss[:fine_tune_start])
            post_fine_tune_best = min(val_loss[fine_tune_start:])
            fine_tuning_improvement = pre_fine_tune_best - post_fine_tune_best

        convergence_metrics.update(
            {
                "is_overfitting": is_overfitting,
                "best_val_loss_epoch": best_val_loss_epoch,
                "epochs_since_best_val_loss": epochs_since_best,
                "has_converged": has_converged,
                "val_loss_stability": val_loss_std_last_10,
                "convergence_final_val_loss": val_loss[
                    -1
                ],  # Renombrado para evitar duplicado
                "best_val_loss": min(val_loss),
                "fine_tuning_improvement": fine_tuning_improvement,
            }
        )

        if verbose:
            status_color = Fore.RED if is_overfitting else Fore.GREEN
            print(
                f"   • Overfitting detectado: {status_color}{'Sí' if is_overfitting else 'No'}{Style.RESET_ALL}"
            )
            print(f"   • Mejor epoch (val_loss): {best_val_loss_epoch}")
            print(f"   • Epochs desde mejor: {epochs_since_best}")
            print(f"   • Ha convergido: {'Sí' if has_converged else 'No'}")
            print(f"   • Estabilidad val_loss: {val_loss_std_last_10:.4f}")
            if fine_tuning_improvement != 0:
                improvement_color = (
                    Fore.GREEN if fine_tuning_improvement > 0 else Fore.RED
                )
                print(
                    f"   • Mejora por fine-tuning: {improvement_color}{fine_tuning_improvement:.4f}{Style.RESET_ALL}"
                )

    # Análisis de gap train-validation
    if train_loss and val_loss and len(train_loss) == len(val_loss):
        final_train_val_gap = val_loss[-1] - train_loss[-1]
        avg_train_val_gap = np.mean([v - t for v, t in zip(val_loss, train_loss)])

        convergence_metrics.update(
            {
                "final_train_val_gap": final_train_val_gap,
                "avg_train_val_gap": avg_train_val_gap,
                "large_gap_warning": final_train_val_gap > 0.5,
            }
        )

        if verbose:
            gap_color = Fore.YELLOW if final_train_val_gap > 0.5 else Fore.GREEN
            print(
                f"   • Gap train-val final: {gap_color}{final_train_val_gap:.4f}{Style.RESET_ALL}"
            )
            print(f"   • Gap train-val promedio: {avg_train_val_gap:.4f}")

    if verbose:
        print("=" * 60)

    return convergence_metrics
