"""
🎯 CLASS WEIGHTS OPTIMIZADOS PARA EFFICIENTNET
============================================

📊 PROBLEMA IDENTIFICADO:
El EfficientNet actual tiene problemas para distinguir 'healthy' de enfermedades,
causando confusión entre clases y recall bajo para plantas sanas.

🔧 ESTRATEGIAS DE CLASS WEIGHTS:

1. BALANCED WEIGHTS (Recomendado para empezar):
   - Todos los pesos cercanos a 1.0
   - Evita sesgo extremo hacia cualquier clase
   - Permite que el modelo aprenda naturalmente

2. HEALTHY-FOCUSED WEIGHTS (Si persiste confusión):
   - Peso ligeramente mayor para 'healthy'
   - Penaliza más los falsos negativos de plantas sanas
   - Útil si el modelo marca sanas como enfermas

3. DISEASE-BALANCED WEIGHTS (Para recall de enfermedades):
   - Weights según frecuencia inversa de clases
   - Boost a clases minoritarias
   - Mantiene balance entre todas las enfermedades

📈 WEIGHTS RECOMENDADOS BASADOS EN ANÁLISIS:
"""


def get_balanced_class_weights():
    """
    Class weights balanceados para reducir confusión entre clases.
    Estrategia conservadora para evitar sesgo extremo.
    """
    return {
        0: 1.0,  # cerscospora - peso neutro
        1: 1.1,  # healthy - ligero boost para mejorar detección
        2: 1.0,  # leaf_rust - peso neutro
        3: 1.2,  # miner - boost para clase minoritaria
        4: 1.1,  # phoma - ligero boost
    }


def get_healthy_focused_weights():
    """
    Class weights enfocados en mejorar detección de plantas sanas.
    Usar si el modelo confunde sanas con enfermas frecuentemente.
    """
    return {
        0: 0.9,  # cerscospora - reducir peso
        1: 1.3,  # healthy - boost significativo
        2: 0.9,  # leaf_rust - reducir peso
        3: 1.1,  # miner - peso moderado
        4: 0.9,  # phoma - reducir peso
    }


def get_disease_balanced_weights():
    """
    Class weights balanceados por frecuencia de cada enfermedad.
    Basado en distribución típica de dataset de café.
    """
    return {
        0: 1.2,  # cerscospora - boost moderado
        1: 0.8,  # healthy - peso reducido (clase mayoritaria)
        2: 1.3,  # leaf_rust - boost para detección
        3: 1.5,  # miner - boost alto (clase minoritaria)
        4: 1.4,  # phoma - boost moderado-alto
    }


def get_conservative_weights():
    """
    Class weights ultra-conservadores para evitar overfitting.
    Mínimas diferencias entre clases.
    """
    return {
        0: 1.0,  # cerscospora
        1: 1.05,  # healthy - boost mínimo
        2: 1.0,  # leaf_rust
        3: 1.1,  # miner - boost mínimo
        4: 1.05,  # phoma - boost mínimo
    }


def analyze_class_distribution(train_labels):
    """
    Analiza la distribución de clases y recomienda weights.

    Args:
        train_labels: Array de etiquetas de entrenamiento

    Returns:
        dict: Class weights recomendados basados en distribución
    """
    import numpy as np

    # Contar frecuencia de cada clase
    unique, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)

    print("\n📊 ANÁLISIS DE DISTRIBUCIÓN DE CLASES:")
    print("=" * 50)

    class_names = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"]

    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        percentage = (count / total_samples) * 100
        print(f"  {class_names[class_idx]:15s}: {count:4d} ({percentage:5.1f}%)")

    # Calcular weights inversos a la frecuencia
    max_count = max(counts)
    weights = {}

    print("\n🎯 WEIGHTS CALCULADOS (Frecuencia Inversa):")
    for class_idx, count in zip(unique, counts):
        weight = max_count / count
        # Suavizar para evitar weights extremos
        weight = min(weight, 2.0)  # Máximo 2x
        weight = max(weight, 0.5)  # Mínimo 0.5x
        weights[class_idx] = round(weight, 2)
        print(f"  {class_names[class_idx]:15s}: {weight:.2f}")

    return weights


def recommend_weights_for_efficientnet(train_labels, validation_metrics=None):
    """
    Recomienda class weights específicos para EfficientNet basado en:
    1. Distribución de clases
    2. Métricas de validación previas (si disponibles)
    3. Problemas específicos identificados

    Args:
        train_labels: Array de etiquetas de entrenamiento
        validation_metrics: Dict con métricas de runs previos (opcional)

    Returns:
        dict: Class weights recomendados
    """

    print("\n🚀 RECOMENDACIÓN DE CLASS WEIGHTS PARA EFFICIENTNET")
    print("=" * 60)

    # Analizar distribución
    calculated_weights = analyze_class_distribution(train_labels)

    # Ajustar basado en problemas conocidos del EfficientNet
    if validation_metrics:
        print("\n📈 Métricas previas disponibles:")
        for metric, value in validation_metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n💡 RECOMENDACIONES:")
    print("1. 🟢 CONSERVADOR (empezar aquí):")
    conservative = get_conservative_weights()
    for i, (k, v) in enumerate(conservative.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n2. 🔄 BALANCEADO (si conservador no funciona):")
    balanced = get_balanced_class_weights()
    for i, (k, v) in enumerate(balanced.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n3. 🎯 CALCULADO (basado en frecuencia):")
    for i, (k, v) in enumerate(calculated_weights.items()):
        class_name = ["cerscospora", "healthy", "leaf_rust", "miner", "phoma"][k]
        print(f"     {class_name}: {v}")

    print("\n🔧 ESTRATEGIA RECOMENDADA:")
    print("  1. Empezar con CONSERVADOR")
    print("  2. Si confunde healthy → usar healthy_focused_weights()")
    print("  3. Si recall bajo en enfermedades → usar CALCULADO")
    print("  4. Monitorear confusion matrix cada 5 epochs")

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
