"""Class weights recommendations for EfficientNet.

This module provides several pre-defined class-weight strategies and a small
analysis helper that computes weights from the training label distribution.

Strategies included:
- conservative: minimal differences between classes to avoid overfitting
- balanced: near-uniform weights to avoid bias
- healthy-focused: higher weight for the "healthy" class
- disease-balanced: inverse-frequency weights to boost rare classes

The recommendations are intended to be used with Keras' ``model.fit`` via
the ``class_weight`` parameter.
"""


def get_balanced_class_weights():
    """Return near-uniform class weights to avoid introducing bias.

    Use this conservative strategy as a neutral starting point.
    """
    return {
        0: 1.0,  # cerscospora - peso neutro
        1: 1.1,  # healthy - ligero boost para mejorar detección
        2: 1.0,  # leaf_rust - peso neutro
        3: 1.2,  # miner - boost para clase minoritaria
        4: 1.1,  # phoma - ligero boost
    }


def get_healthy_focused_weights():
    """Return class weights that prioritize correct healthy predictions.

    Use when the model frequently confuses healthy plants with diseased ones.
    """
    return {
        0: 0.9,  # cerscospora - reducir peso
        1: 1.3,  # healthy - boost significativo
        2: 0.9,  # leaf_rust - reducir peso
        3: 1.1,  # miner - peso moderado
        4: 0.9,  # phoma - reducir peso
    }


def get_disease_balanced_weights():
    """Return class weights based on inverse-frequency boosting.

    This strategy increases weights for under-represented disease classes.
    """
    return {
        0: 1.2,  # cerscospora - boost moderado
        1: 0.8,  # healthy - peso reducido (clase mayoritaria)
        2: 1.3,  # leaf_rust - boost para detección
        3: 1.5,  # miner - boost alto (clase minoritaria)
        4: 1.4,  # phoma - boost moderado-alto
    }


def get_conservative_weights():
    """Return conservative weights with minimal differences between classes.

    This reduces the risk of overfitting caused by large weight disparities.
    """
    return {
        0: 1.0,  # cerscospora
        1: 1.05,  # healthy - boost mínimo
        2: 1.0,  # leaf_rust
        3: 1.1,  # miner - boost mínimo
        4: 1.05,  # phoma - boost mínimo
    }


def analyze_class_distribution(train_labels):
    """Analyze class frequency and compute inverse-frequency weights.

    Args:
        train_labels: Array-like of training labels.

    Returns:
        dict: Mapping from class index to recommended weight (float).
    """
    import numpy as np

    # Count occurrences per class
    unique, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)

    print("\n📊 ANÁLISIS DE DISTRIBUCIÓN DE CLASES:")
    print("=" * 50)

    class_names = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"]

    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        percentage = (count / total_samples) * 100
        print(f"  {class_names[class_idx]:15s}: {count:4d} ({percentage:5.1f}%)")

    # Compute inverse-frequency weights and clip extremes
    max_count = max(counts)
    weights = {}

    print("\n🎯 WEIGHTS CALCULADOS (Frecuencia Inversa):")
    for class_idx, count in zip(unique, counts):
        weight = max_count / count
        # Smooth to avoid extreme weights
        weight = min(weight, 2.0)  # max 2x
        weight = max(weight, 0.5)  # min 0.5x
        weights[class_idx] = round(weight, 2)
        print(f"  {class_names[class_idx]:15s}: {weight:.2f}")

    return weights


def recommend_weights_for_efficientnet(train_labels, validation_metrics=None):
    """Recommend class weights for EfficientNet.

    The recommendation considers class distribution and optional prior
    validation metrics. Returns a conservative default weight mapping.
    """

    print("\n🚀 RECOMENDACIÓN DE CLASS WEIGHTS PARA EFFICIENTNET")
    print("=" * 60)

    # Analyze distribution and compute frequency-based weights
    calculated_weights = analyze_class_distribution(train_labels)

    # If validation metrics are provided, display them for context
    if validation_metrics:
        print("\n📈 Métricas previas disponibles:")
        for metric, value in validation_metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n💡 RECOMMENDATIONS:")
    print("1. 🟢 CONSERVATIVE (start here):")
    conservative = get_conservative_weights()
    for i, (k, v) in enumerate(conservative.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n2. 🔄 BALANCED (if conservative does not work):")
    balanced = get_balanced_class_weights()
    for i, (k, v) in enumerate(balanced.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n3. 🎯 CALCULATED (based on frequency):")
    for i, (k, v) in enumerate(calculated_weights.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n🔧 RECOMMENDED STRATEGY:")
    print("  1. Start with CONSERVATIVE")
    print("  2. If healthy is confused → use healthy_focused_weights()")
    print("  3. If disease recall is low → use CALCULATED")
    print("  4. Monitor confusion matrix every 5 epochs")

    return conservative  # Devolver weights conservadores por defecto


if __name__ == "__main__":
    print("🎯 CLASS WEIGHTS OPTIMIZER FOR EFFICIENTNET")
    print("=" * 50)
    print("\nEste archivo contiene estrategias optimizadas de class weights")
    print("para resolver problemas de confusión entre clases en EfficientNet.")
    print("\nUso recomendado:")
    print("1. from class_weights_optimizer import recommend_weights_for_efficientnet")
    print("2. weights = recommend_weights_for_efficientnet(train_labels)")
    print("3. Usar weights en model.fit(class_weight=weights)")
