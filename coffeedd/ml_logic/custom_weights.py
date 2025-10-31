import numpy as np
from coffeedd.ml_logic.data_analysis import find_rarest_disease_class

from coffeedd.params import LOCAL_DATA_PATH, CLASS_NAMES, NUM_CLASSES

def get_class_weights(
    train_labels,
):
    ## Calcular class weights dinámicos basados en distribución real del train set
    print("\n" + "="*60)
    print("⚖️  CALCULANDO CLASS WEIGHTS DINÁMICOS")
    print("="*60)

    # Encontrar la clase más rara con análisis de desequilibrio
    rarest_disease, rarest_count, is_extremely_rare, class_stats = find_rarest_disease_class(
        LOCAL_DATA_PATH, CLASS_NAMES, extreme_threshold=0.5
    )

    if rarest_disease:
        print("\n📈 Análisis de distribución de clases en dataset completo:")
        for class_name, count in class_stats['all_counts'].items():
            percentage = (count / class_stats['avg_count']) * 100
            emoji = "⚠️" if class_name == rarest_disease else "🦠"
            print(f"  {emoji} {class_name:15s}: {count:5d} muestras ({percentage:5.1f}% vs promedio)")

        print(f"\n  📉 Clase más pequeña: {rarest_disease} ({rarest_count} muestras)")
        print(f"  📊 Promedio por clase: {class_stats['avg_count']:.1f} muestras")
        print(f"  🔢 Ratio vs promedio: {class_stats['ratio_vs_avg']:.2f}")
        print(f"  🔢 Ratio vs máxima: {class_stats['ratio_vs_max']:.2f}")

        if is_extremely_rare:
            print(f"  🚨 DESEQUILIBRIO EXTREMO detectado - Activando boost especial para {rarest_disease}")
        else:
            print("  ✅ Distribución relativamente balanceada - No se requiere boost especial")

    # Calcular class weights con distribución actual del train set
    from sklearn.utils.class_weight import compute_class_weight

    # Calcular pesos balanceados automáticamente
    unique_classes = np.unique(train_labels)
    class_weights_dict = {}

    if len(unique_classes) > 1:
        # Usar sklearn para calcular pesos balanceados
        computed_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_labels
        )

        for i, class_idx in enumerate(unique_classes):
            class_weights_dict[class_idx] = computed_weights[i]

    print(f"\n📊 Distribución en train set ({len(train_labels)} muestras):")
    for idx in range(NUM_CLASSES):
        if idx in class_weights_dict:
            count = np.sum(train_labels == idx)
            percentage = (count / len(train_labels)) * 100
            print(f"  {CLASS_NAMES[idx]:15s}: {count:4d} ({percentage:5.1f}%) - peso: {class_weights_dict[idx]:.3f}")

    # Ajustes manuales para priorizar detección de enfermedades
    class_weights = {}

    # Detectar si estamos usando una muestra pequeña
    is_small_sample = len(train_labels) < 5000

    for idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[idx]

        # Si la clase existe en train, aplicar ajustes
        if idx in class_weights_dict:
            weight = class_weights_dict[idx]

            # Con muestras pequeñas, usar ajustes MUY conservadores
            if class_name == 'healthy':
                # PENALIZAR FUERTEMENTE healthy para evitar falsos negativos
                if is_small_sample:
                    class_weights[idx] = max(weight * 0.3, 0.1)  # Muy penalizado
                else:
                    class_weights[idx] = weight * 0.9  # Penalizado (antes 0.8)

            elif class_name == rarest_disease and is_extremely_rare:
                # BOOST MÁXIMO solo para clases con desequilibrio extremo
                if is_small_sample:
                    class_weights[idx] = min(weight * 3.0, 8.0)
                else:
                    class_weights[idx] = weight * 2.0  # Boost fuerte para desequilibrio extremo
                print(f"  🚨 Aplicando boost extremo a {class_name} (desequilibrio {class_stats['ratio_vs_avg']:.2f})")

            else:
                # BOOST MODERADO para otras enfermedades (sin boost especial)
                if is_small_sample:
                    class_weights[idx] = min(weight * 1.3, 3.0)  # Reducido de 1.8
                else:
                    class_weights[idx] = weight * 1.1  # Reducido de 1.2
        else:
            # Clases ausentes - peso alto para enfermedades
            if class_name == 'healthy':
                class_weights[idx] = 0.5 if is_small_sample else 0.8
            else:
                class_weights[idx] = 5.0 if is_small_sample else 8.0

    print("\n📊 Class weights FINALES aplicados para RECALL:")
    for idx, class_name in enumerate(CLASS_NAMES):
        status = "✓" if idx in class_weights_dict else "⚠️ (ausente)"

        # Emojis inteligentes
        if class_name == 'healthy':
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

        print(f"  {emoji} {class_name:20s}: {class_weights[idx]:6.3f} {status} {priority_note}")

    # IMPORTANTE: Convertir class_weights a floats de Python para evitar errores de serialización
    class_weights = {k: float(v) for k, v in class_weights.items()}
    print("\n✅ Class weights convertidos a formato serializable")

    return class_weights
