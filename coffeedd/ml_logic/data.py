import os
from collections import Counter
import numpy as np
import tensorflow as tf

from coffeedd.params import NUM_CLASSES
from coffeedd.ml_logic.preprocessing import parse_image, augment_image


def create_dataset_from_directory(
    data_path,
    class_names,
    validation_split=0.15,
    test_split=0.15,
    seed=42,
    sample_size=None,
):
    """
    Crea datasets de entrenamiento, validación y test de forma eficiente.
    Carga imágenes on-the-fly sin llenar la RAM.

    Args:
        data_path: Ruta al directorio con las carpetas de clases
        class_names: Lista de nombres de clases
        validation_split: Proporción para validación (default 0.15 = 15%)
        test_split: Proporción para test (default 0.15 = 15%)
        seed: Semilla para reproducibilidad
        sample_size: Número de imágenes a usar. Opciones:
            - None/'full': Usa todas las imágenes
            - int (ej: 100, 1000): Número exacto de imágenes totales
            - 'half': Usa la mitad del dataset
            - float (ej: 0.25): Usa ese porcentaje del dataset
    """

    # 1. Recopilar todas las rutas de imágenes y sus etiquetas
    print("📂 Escaneando directorio...")
    image_paths = []
    labels = []
    class_counts = {}

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Advertencia: No se encontró la carpeta {class_name}")
            continue

        # Obtener todas las imágenes .jpg
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        image_paths.extend(images)
        labels.extend([class_idx] * len(images))
        class_counts[class_name] = len(images)

        print(f"  {class_name}: {len(images)} imágenes")

    total_images = len(image_paths)
    print(f"\n✅ Total disponible: {total_images} imágenes")

    # 2. Aplicar muestreo si se especifica
    if sample_size is not None and sample_size != "full":
        print("\n🎲 Aplicando muestreo...")

        # Convertir a arrays para manipular
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Determinar número de imágenes a usar
        if sample_size == "half":
            target_size = total_images // 2
        elif float(sample_size) < 1.0:  # Es un float (porcentaje)
            target_size = int(total_images * float(sample_size))
        else:  # Es un número entero
            target_size = min(int(sample_size), int(total_images))

        # Muestreo estratificado (mantiene proporción de clases)
        from sklearn.model_selection import train_test_split

        if target_size < total_images:
            _, image_paths, _, labels = train_test_split(
                image_paths,
                labels,
                test_size=target_size / total_images,
                stratify=labels,
                random_state=seed,
            )

            print(f"  📉 Reducido de {total_images} a {len(image_paths)} imágenes")
            print("  📊 Distribución muestreada:")
            for class_idx, class_name in enumerate(class_names):
                count = np.sum(labels == class_idx)
                percentage = (count / len(labels)) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
    else:
        # Convertir a arrays numpy
        image_paths = np.array(image_paths)
        labels = np.array(labels)

    # 3. Shuffle con seed fijo para reproducibilidad
    np.random.seed(seed)
    indices = np.random.permutation(len(image_paths))
    image_paths = image_paths[indices]
    labels = labels[indices]

    # 4. Dividir en train/val/test
    total_size = len(image_paths)
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)
    # train_size = total_size - test_size - val_size

    # Test set
    test_paths = image_paths[:test_size]
    test_labels = labels[:test_size]

    # Validation set
    val_paths = image_paths[test_size : test_size + val_size]
    val_labels = labels[test_size : test_size + val_size]

    # Train set
    train_paths = image_paths[test_size + val_size :]
    train_labels = labels[test_size + val_size :]

    print("\n📊 División de datos:")
    print(f"  Train: {len(train_paths)} ({len(train_paths)/total_size*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} ({len(val_paths)/total_size*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} ({len(test_paths)/total_size*100:.1f}%)")

    # 5. Mostrar distribución por clase en cada set
    print("\n📈 Distribución por clase:")
    for set_name, set_labels in [
        ("Train", train_labels),
        ("Val", val_labels),
        ("Test", test_labels),
    ]:
        counts = Counter(set_labels)
        print(f"\n  {set_name}:")
        for class_idx, class_name in enumerate(class_names):
            count = counts.get(class_idx, 0)
            percentage = (count / len(set_labels) * 100) if len(set_labels) > 0 else 0
            print(f"    {class_name:20s}: {count:5d} ({percentage:5.1f}%)")

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def create_tf_dataset(
    image_paths, labels, batch_size, is_training=False, augment=False
):
    """
    Crea un tf.data.Dataset optimizado que carga imágenes on-the-fly
    """
    # Crear dataset desde paths y labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if is_training:
        # Shuffle solo en training
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    # Cargar imágenes (en paralelo)
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Aplicar augmentation si es necesario
    if augment and is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Convertir labels a one-hot
    dataset = dataset.map(
        lambda img, lbl: (img, tf.one_hot(lbl, NUM_CLASSES)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Batch
    dataset = dataset.batch(batch_size)

    # Prefetch para optimizar performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
