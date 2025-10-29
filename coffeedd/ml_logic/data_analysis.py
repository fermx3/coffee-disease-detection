import os

from coffeedd.params import CLASS_NAMES, EPOCHS

def find_rarest_disease_class(data_path, class_names, extreme_threshold=0.5):
    """
    Encuentra la clase de enfermedad con menos muestras solo si hay un desequilibrio extremo

    Args:
        data_path: Ruta al dataset
        class_names: Lista de nombres de clases
        extreme_threshold: Factor mínimo de diferencia para considerar "extremadamente rara"
                          (ej: 0.5 = la clase rara debe tener menos de la mitad que el promedio)

    Returns:
        tuple: (clase_más_rara, count, es_extrema, estadísticas)
    """
    # Detectar automáticamente las clases de enfermedad (todas excepto 'healthy')
    disease_classes = [name for name in CLASS_NAMES if name != 'healthy']
    num_disease_classes = len(disease_classes)
    class_counts = {}

    for class_name in class_names:
        if class_name == 'healthy':
            continue

        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count

    if not class_counts:
        return None, 0, False, {}

    # Calcular estadísticas
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)

    # Encontrar la clase más rara
    rarest_class = min(class_counts, key=class_counts.get)

    # Determinar si es extremadamente rara
    # Criterios múltiples para considerar "extrema":
    ratio_vs_avg = min_count / avg_count if avg_count > 0 else 1
    ratio_vs_max = min_count / max_count if max_count > 0 else 1

    # Es extrema si tiene menos del threshold% comparado con el promedio
    # Y menos del 30% comparado con la clase más común
    is_extreme = (ratio_vs_avg < extreme_threshold and ratio_vs_max < 0.3)

    stats = {
        'all_counts': class_counts,
        'min_count': min_count,
        'max_count': max_count,
        'avg_count': avg_count,
        'ratio_vs_avg': ratio_vs_avg,
        'ratio_vs_max': ratio_vs_max,
        'is_extreme': is_extreme
    }


    print("\n🎯 Configuración optimizada para DETECCIÓN DE ENFERMEDADES:")
    print("  📊 Objetivo: Minimizar falsos negativos (enfermedad → healthy)")
    print(f"  🦠 Clases de enfermedad detectadas: {num_disease_classes} ({', '.join(disease_classes)})")
    print("  🌱 Clase healthy: Será penalizada para evitar falsos negativos")
    print("  📈 Métrica principal: Recall")
    print(f"  ⏰ Epochs configurados: {EPOCHS}")
    print("\n⏳ Class weights se calcularán después de cargar los datos...")

    return rarest_class, min_count, is_extreme, stats

def false_negatives_analysis(
    test_labels,
    y_pred_test_classes
):
    """
    Realiza un análisis detallado de los falsos negativos en el conjunto de prueba.
    Args:
        test_labels: Etiquetas reales del conjunto de prueba.
        y_pred_test_classes: Predicciones del modelo para el conjunto de prueba.
    """
    import numpy as np
    print("\n" + "="*60)
    print("⚠️  ANÁLISIS DETALLADO DE FALSOS NEGATIVOS")
    print("="*60)

    healthy_idx = CLASS_NAMES.index('healthy')

    print("\n⚠️  Falsos Negativos (Enfermedad → Healthy):")

    total_fn = 0
    total_disease_samples = 0

    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name == 'healthy':
            continue  # Saltar la clase healthy

        # Máscara para casos reales de esta enfermedad
        mask_true = (test_labels == idx)
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            # Máscara para predicciones incorrectas como 'healthy'
            mask_pred_healthy = (y_pred_test_classes == healthy_idx)

            # Falsos negativos: casos reales de enfermedad predichos como healthy
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

    # Análisis adicional: ¿A qué clases se confunden las enfermedades?
    print("\n🔍 Análisis de confusiones por enfermedad:")
    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name == 'healthy':
            continue

        mask_true = (test_labels == idx)
        total_cases = np.sum(mask_true)

        if total_cases > 0:
            print(f"\n  {class_name} (total: {total_cases}):")
            predictions_for_this_class = y_pred_test_classes[mask_true]

            for pred_idx, pred_class in enumerate(CLASS_NAMES):
                count = np.sum(predictions_for_this_class == pred_idx)
                if count > 0:
                    percentage = (count / total_cases) * 100
                    emoji = "✅" if pred_idx == idx else ("❌" if pred_class == 'healthy' else "🔄")
                    print(f"    {emoji} → {pred_class:15s}: {count}/{total_cases} ({percentage:.1f}%)")
