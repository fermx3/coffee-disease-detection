"""Dynamic class-weight calculator tuned for coffee disease datasets.

This module inspects dataset-level imbalance (via
``find_rarest_disease_class``) and combines that information with
``sklearn.utils.class_weight.compute_class_weight`` computed weights to
produce a final, serializable mapping usable in Keras' ``model.fit``
``class_weight`` argument.

User-facing prints are intentionally left in Spanish to match the
project's CLI messaging.
"""

import numpy as np
from coffeedd.ml_logic.data_analysis import find_rarest_disease_class

from coffeedd.params import LOCAL_DATA_PATH, CLASS_NAMES, NUM_CLASSES


def get_class_weights(train_labels):
    """Compute final class weights based on train labels and dataset stats.

    Steps:
    1. Run a dataset-level imbalance analysis (find_rarest_disease_class).
    2. Compute sklearn 'balanced' weights from the train labels.
    3. Apply conservative manual adjustments for small samples and extreme
       imbalance cases.

    Args:
        train_labels: array-like of integer class labels from the training set.

    Returns:
        dict mapping class index -> float weight (serializable Python floats).
    """

    # Informational header (kept in Spanish intentionally)
    print("\n" + "=" * 60)
    print("⚖️  CALCULANDO CLASS WEIGHTS DINÁMICOS")
    print("=" * 60)

    # Run dataset-level rarity analysis
    rarest_disease, rarest_count, is_extremely_rare, class_stats = (
        find_rarest_disease_class(LOCAL_DATA_PATH, CLASS_NAMES, extreme_threshold=0.5)
    )

    if rarest_disease:
        print("\n📈 Análisis de distribución de clases en dataset completo:")
        for class_name, count in class_stats["all_counts"].items():
            percentage = (count / class_stats["avg_count"]) * 100
            emoji = "⚠️" if class_name == rarest_disease else "🦠"
            print(
                f"  {emoji} {class_name:15s}: {count:5d} muestras ({percentage:5.1f}% vs promedio)"
            )

        print(f"\n  📉 Clase más pequeña: {rarest_disease} ({rarest_count} muestras)")
        print(f"  📊 Promedio por clase: {class_stats['avg_count']:.1f} muestras")
        print(f"  🔢 Ratio vs promedio: {class_stats['ratio_vs_avg']:.2f}")
        print(f"  🔢 Ratio vs máxima: {class_stats['ratio_vs_max']:.2f}")

        if is_extremely_rare:
            print(
                f"  🚨 DESEQUILIBRIO EXTREMO detectado - Activando boost especial para {rarest_disease}"
            )
        else:
            print(
                "  ✅ Distribución relativamente balanceada - No se requiere boost especial"
            )

    # Compute sklearn balanced weights from train labels
    from sklearn.utils.class_weight import compute_class_weight

    unique_classes = np.unique(train_labels)
    class_weights_dict = {}

    if len(unique_classes) > 1:
        computed_weights = compute_class_weight(
            "balanced", classes=unique_classes, y=train_labels
        )

        for i, class_idx in enumerate(unique_classes):
            class_weights_dict[class_idx] = computed_weights[i]

    # Print train-set distribution summary
    print(f"\n📊 Distribución en train set ({len(train_labels)} muestras):")
    for idx in range(NUM_CLASSES):
        if idx in class_weights_dict:
            count = np.sum(train_labels == idx)
            percentage = (count / len(train_labels)) * 100
            print(
                f"  {CLASS_NAMES[idx]:15s}: {count:4d} ({percentage:5.1f}%) - peso: {class_weights_dict[idx]:.3f}"
            )

    # Manual adjustments to favor disease recall while avoiding extremes
    class_weights = {}
    is_small_sample = len(train_labels) < 5000

    for idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[idx]

        if idx in class_weights_dict:
            weight = class_weights_dict[idx]

            # Very conservative adjustments for small samples
            if class_name == "healthy":
                # Penalize 'healthy' to reduce false negatives of diseased samples
                if is_small_sample:
                    class_weights[idx] = max(weight * 0.3, 0.1)
                else:
                    class_weights[idx] = weight * 0.9

            elif class_name == rarest_disease and is_extremely_rare:
                # Strong boost for extreme rarity
                if is_small_sample:
                    class_weights[idx] = min(weight * 3.0, 8.0)
                else:
                    class_weights[idx] = weight * 2.0
                print(
                    f"  🚨 Aplicando boost extremo a {class_name} (desequilibrio {class_stats['ratio_vs_avg']:.2f})"
                )

            else:
                # Moderate boost for other diseases
                if is_small_sample:
                    class_weights[idx] = min(weight * 1.3, 3.0)
                else:
                    class_weights[idx] = weight * 1.0
        else:
            # Absent classes: assign high weight for diseases to encourage recall
            if class_name == "healthy":
                class_weights[idx] = 0.5 if is_small_sample else 0.8
            else:
                class_weights[idx] = 5.0 if is_small_sample else 8.0

    # Final summary (kept in Spanish)
    print("\n📊 Class weights FINALES aplicados para RECALL:")
    for idx, class_name in enumerate(CLASS_NAMES):
        status = "✓" if idx in class_weights_dict else "⚠️ (ausente)"

        if class_name == "healthy":
            emoji = "🌱"
            priority_note = "(penalizada)"
        elif class_name == rarest_disease and is_extremely_rare:
            emoji = "🚨"
            priority_note = "(desequilibrio extremo - boost máximo)"
        elif class_name == rarest_disease:
            emoji = "⚠️"
            priority_note = "(más pequeña pero balanceada)"
        else:
            emoji = "🦠"
            priority_note = ""

        print(
            f"  {emoji} {class_name:20s}: {class_weights[idx]:6.3f} {status} {priority_note}"
        )

    # Convert to plain Python floats for safe serialization
    class_weights = {k: float(v) for k, v in class_weights.items()}
    print("\n✅ Class weights convertidos a formato serializable")

    return class_weights
