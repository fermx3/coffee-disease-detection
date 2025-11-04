import os
import time
import tempfile
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from colorama import Fore, Style
from google.cloud import storage

import keras
from coffeedd.params import MODEL_TARGET, LOCAL_REGISTRY_PATH, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, BUCKET_NAME, MODEL_ARCHITECTURE

def find_latest_model_by_architecture(base_dir, target_architecture=None):
        """Encuentra el modelo más reciente, opcionalmente filtrado por arquitectura"""
        models_found = []

        # Si se especifica una arquitectura, buscar solo en esa carpeta
        if target_architecture:
            arch_dir = os.path.join(base_dir, target_architecture)
            if os.path.exists(arch_dir):
                search_dirs = [arch_dir]
            else:
                return None
        else:
            # Buscar en todas las carpetas de arquitectura + directorio base (compatibilidad)
            search_dirs = [base_dir]
            for arch in ['cnn', 'vgg16', 'efficientnet']:
                arch_dir = os.path.join(base_dir, arch)
                if os.path.exists(arch_dir):
                    search_dirs.append(arch_dir)

        # Buscar archivos en cada directorio
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            items = os.listdir(search_dir)

            # Archivos de modelo (prioridad alta)
            # Priorizar archivos .weights.h5 sobre .keras vacíos
            model_files = []
            for ext in ['.weights.h5', '.keras', '.h5']:  # .weights.h5 primero
                files = [
                    os.path.join(search_dir, item)
                    for item in items
                    if item.endswith(ext)
                ]
                # Filtrar archivos muy pequeños que probablemente estén corruptos
                valid_files = []
                for file_path in files:
                    try:
                        file_size = os.path.getsize(file_path)
                        # Archivos .keras/.h5 deben ser > 1MB, .weights.h5 > 5MB
                        min_size = 5 * 1024 * 1024 if file_path.endswith('.weights.h5') else 1024 * 1024
                        if file_size > min_size:
                            valid_files.append(file_path)
                    except:
                        continue
                model_files.extend(valid_files)

            models_found.extend(model_files)

            # Solo buscar directorios SavedModel si no hay archivos de modelo
            # y no estamos en un directorio de arquitectura
            if not models_found and search_dir == base_dir:
                # Directorios SavedModel válidos (deben contener saved_model.pb)
                for item in items:
                    item_path = os.path.join(search_dir, item)
                    if (os.path.isdir(item_path) and
                        not item.startswith('.') and
                        not item in ['cnn', 'vgg16', 'efficientnet'] and  # Excluir carpetas de arquitectura
                        os.path.exists(os.path.join(item_path, 'saved_model.pb'))):  # Validar SavedModel
                        models_found.append(item_path)

        if not models_found:
            return None

        # Retornar el más reciente
        return max(models_found, key=lambda p: os.path.getmtime(p))

def detect_model_architecture(model_path_or_layers):
    """Detecta la arquitectura del modelo desde archivo o capas"""
    if isinstance(model_path_or_layers, str):
        # Detectar desde nombre de archivo
        filename = model_path_or_layers.lower()
        if 'efficientnet' in filename:
            return 'efficientnet'
        elif 'vgg16' in filename:
            return 'vgg16'
        elif 'cnn' in filename:
            return 'cnn'
        else:
            return MODEL_ARCHITECTURE.lower()  # Default desde params
    else:
        # Detectar desde capas del modelo
        layers = model_path_or_layers
        layer_names = [layer.name.lower() for layer in layers]

        # Detectar EfficientNet
        for layer_name in layer_names:
            if 'efficientnet' in layer_name:
                return 'efficientnet'

        # Detectar VGG16 - método 1: capa funcional llamada "vgg16"
        if 'vgg16' in layer_names:
            return 'vgg16'

        # Detectar VGG16 - método 2: patrones característicos de capas individuales
        vgg16_patterns = [
            'block1_conv1', 'block1_conv2', 'block1_pool',
            'block2_conv1', 'block2_conv2', 'block2_pool',
            'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
            'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool',
            'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool'
        ]

        # Si encuentra al menos 5 patrones VGG16, es VGG16
        vgg16_matches = sum(1 for pattern in vgg16_patterns if any(pattern in name for name in layer_names))
        if vgg16_matches >= 5:
            return 'vgg16'

        # Detectar VGG16 - método 3: presencia de GlobalAveragePooling2D + estructura típica VGG16
        has_global_avg_pool = any('global_average_pooling2d' in name for name in layer_names)
        has_batch_norm = any('batch_normalization' in name for name in layer_names)
        has_multiple_dense = len([name for name in layer_names if 'dense' in name]) >= 2

        # Si tiene GlobalAveragePooling2D + BatchNorm + múltiples Dense = probablemente VGG16 optimizado
        if has_global_avg_pool and has_batch_norm and has_multiple_dense:
            return 'vgg16'

        # Detectar CNN simple por ausencia de transfer learning y pocas capas
        # CNN simple típicamente tiene <10 capas
        if len(layers) < 10:
            return 'cnn'

        # Si tiene muchas capas pero no es VGG16 ni EfficientNet, probablemente sea CNN complejo
        return 'cnn'

def build_model_by_architecture(architecture):
    """Construye el modelo según la arquitectura especificada"""
    from coffeedd.ml_logic.model import build_simple_cnn_model, build_efficientnet_model, build_vgg16_model

    arch = architecture.lower()
    if arch == 'efficientnet':
        model, _ = build_efficientnet_model()  # Desempaquetar tupla, solo devolver modelo
        return model
    elif arch == 'vgg16':
        return build_vgg16_model()
    elif arch == 'cnn':
        return build_simple_cnn_model()
    else:
        print(f"⚠️  Arquitectura desconocida '{architecture}', usando CNN por defecto")
        return build_simple_cnn_model()

def load_model_by_architecture(architecture: str, stage="Production", compile_with_metrics=True) -> keras.Model:
    """
    Carga el modelo más reciente de una arquitectura específica

    Args:
        architecture: 'cnn', 'vgg16', o 'efficientnet'
        stage: Stage de MLflow (si aplica)
        compile_with_metrics: Si compilar con métricas completas

    Returns:
        keras.Model: Modelo cargado
    """
    print(f"🎯 Cargando modelo específico de arquitectura: {architecture}")

    if MODEL_TARGET == "local":
        from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric
        from coffeedd.ml_logic.model import build_simple_cnn_model, build_efficientnet_model, build_vgg16_model

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

        # Buscar específicamente en la carpeta de esa arquitectura
        model_path = find_latest_model_by_architecture(local_model_directory, architecture)

        if not model_path:
            print(Fore.YELLOW + f"⚠️  No se encontró modelo de arquitectura {architecture}, usando carga general" + Style.RESET_ALL)
            return load_model(stage, compile_with_metrics)

        print(f"📁 Modelo encontrado: {model_path}")
        # Usar la misma lógica de carga que load_model pero con el archivo específico
        # [Se podría implementar la lógica de carga aquí o reutilizar]

    # Para otros targets, usar la función general
    return load_model(stage, compile_with_metrics)

def load_model(stage="Production", compile_with_metrics=True) -> keras.Model:
    from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric
    from coffeedd.ml_logic.model import build_simple_cnn_model, build_efficientnet_model, build_vgg16_model

    def reconstruct_model_from_weights(model_path, architecture):
        """Reconstruye un modelo desde archivos de pesos usando múltiples estrategias"""
        print(f"🏗️  Reconstruyendo modelo {architecture}...")
        print(f"📁 Archivo de pesos: {os.path.basename(model_path)}")

        # Función helper para probar carga de pesos
        def try_load_weights(model, path, strategy_name):
            try:
                print(f"   📊 Modelo tiene {len(model.layers)} capas")
                model.load_weights(path)
                print(Fore.GREEN + f"   ✅ Pesos cargados exitosamente ({strategy_name})" + Style.RESET_ALL)
                return True
            except Exception as e:
                error_msg = str(e)
                if "Layer count mismatch" in error_msg:
                    print(f"   ❌ {strategy_name}: Incompatibilidad de capas")
                elif "axes don't match array" in error_msg:
                    print(f"   ❌ {strategy_name}: Incompatibilidad de dimensiones")
                else:
                    print(f"   ❌ {strategy_name}: {error_msg[:100]}")
                return False

        # Estrategias por arquitectura
        if architecture == 'efficientnet':
            strategies = [
                ("EfficientNet estándar", "standard"),
                ("EfficientNet con fine-tuning", "finetuning")
            ]
        elif architecture == 'vgg16':
            strategies = [("VGG16 estándar", "standard")]
        else:  # CNN simple
            strategies = [("CNN simple", "standard")]

        # Probar cada estrategia
        for strategy_name, strategy_type in strategies:
            try:
                print(f"🔍 Intentando estrategia: {strategy_name}...")

                # Construir modelo según arquitectura
                if architecture == 'efficientnet':
                    model, base_model = build_efficientnet_model()

                    if strategy_type == "finetuning":
                        # Configurar fine-tuning
                        base_model.trainable = True
                        fine_tune_at = len(base_model.layers) - 15
                        for layer in base_model.layers[:fine_tune_at]:
                            layer.trainable = False
                        print(f"   🔧 Fine-tuning configurado: {fine_tune_at} capas congeladas")

                        # Compilar con learning rate ajustado
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=0.001 / 10),
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                    else:
                        # Compilar estándar
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )

                elif architecture == 'vgg16':
                    model = build_vgg16_model()
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                else:  # CNN simple
                    model = build_simple_cnn_model()
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                # Intentar cargar pesos
                if try_load_weights(model, model_path, strategy_name):
                    return model

            except Exception as e:
                print(f"   ❌ Error en construcción del modelo: {e}")
                continue

        raise ValueError(f"No se pudo reconstruir el modelo {architecture} con ninguna estrategia")

    # Función auxiliar para cargar desde local
    def load_from_local():
        print(Fore.MAGENTA + "\n📥 Cargando modelo desde almacenamiento local..." + Style.RESET_ALL)
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

        if not os.path.exists(local_model_directory):
            print(Fore.RED + f"❌ Directorio no existe: {local_model_directory}" + Style.RESET_ALL)
            return None

        # Buscar el modelo más reciente (puede estar en subcarpetas de arquitectura)
        most_recent_model_path_on_disk = find_latest_model_by_architecture(local_model_directory)

        if not most_recent_model_path_on_disk:
            print(Fore.RED + "❌ No se encontraron modelos en el almacenamiento local." + Style.RESET_ALL)
            return None

        try:
            print(f"📂 Cargando: {os.path.basename(most_recent_model_path_on_disk)}")

            # Si es archivo de pesos (.weights.h5), reconstruir el modelo
            if most_recent_model_path_on_disk.endswith('.weights.h5'):
                print(Fore.BLUE + "🔧 Detectado archivo de pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                # Leer la configuración (buscar en el mismo directorio que el archivo de pesos)
                timestamp = os.path.basename(most_recent_model_path_on_disk).replace('.weights.h5', '')
                model_dir = os.path.dirname(most_recent_model_path_on_disk)
                config_path = os.path.join(model_dir, f"{timestamp}_config.json")

                import json
                with open(config_path, 'r') as f:
                    model_info = json.load(f)

                # Detectar arquitectura desde config o por compatibilidad
                if 'architecture' in model_info:
                    architecture = model_info['architecture']
                elif 'model_type' in model_info:
                    # Compatibilidad con configs antiguos
                    model_type = model_info['model_type']
                    if model_type == 'EfficientNet':
                        architecture = 'efficientnet'
                    elif model_type == 'VGG16':
                        architecture = 'vgg16'
                    else:
                        architecture = 'cnn'
                else:
                    architecture = detect_model_architecture(most_recent_model_path_on_disk)

                print(f"🔍 Arquitectura detectada: {architecture}")

                # Usar la función de reconstrucción simplificada
                model = reconstruct_model_from_weights(most_recent_model_path_on_disk, architecture)

            else:
                # Intentar cargar modelo completo (.keras o SavedModel)
                try:
                    model = keras.models.load_model(
                        most_recent_model_path_on_disk,
                        custom_objects={'DiseaseRecallMetric': DiseaseRecallMetric}
                    )

                    architecture = detect_model_architecture(most_recent_model_path_on_disk)

                except ValueError as e:
                    if "No model config found" in str(e):
                        # Es un archivo de solo pesos con extensión .keras
                        print(Fore.YELLOW + "⚠️  Archivo contiene solo pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                        # Detectar tipo por nombre del archivo
                        architecture = detect_model_architecture(most_recent_model_path_on_disk)
                        print(f"🔍 Arquitectura detectada: {architecture}")

                        # Usar la función de reconstrucción simplificada
                        model = reconstruct_model_from_weights(most_recent_model_path_on_disk, architecture)
                    else:
                        raise

            # Recompilar con todas las métricas si se solicita
            if compile_with_metrics:
                print(Fore.BLUE + "🔧 Recompilando con métricas completas..." + Style.RESET_ALL)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.Precision(name='precision'),
                        DiseaseRecallMetric(),
                        keras.metrics.AUC(name='auc')
                    ]
                )

            print(Fore.GREEN + "✅ Modelo cargado exitosamente" + Style.RESET_ALL)
            # Detectar el tipo desde el modelo cargado
            final_architecture = detect_model_architecture(model.layers)
            arch_display_names = {
                'efficientnet': 'EfficientNetB0',
                'vgg16': 'VGG16',
                'cnn': 'CNN simple'
            }
            display_name = arch_display_names.get(final_architecture, final_architecture)
            print(f"🏷️  Tipo: {display_name}")

            return model

        except Exception as e:
            print(Fore.RED + f"❌ Error al cargar modelo local: {e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc()
            return None

    # ============================================================
    # LÓGICA PRINCIPAL CON FALLBACK
    # ============================================================

    if MODEL_TARGET == "local":
        return load_from_local()

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + "\n☁️  Cargando modelo desde Google Cloud Storage..." + Style.RESET_ALL)

        try:
            client = storage.Client()
            bucket = client.get_bucket(BUCKET_NAME)

            # Buscar archivos de modelo en estructura de carpetas de arquitectura
            all_blobs = []

            # Buscar en carpetas de arquitectura específicas
            # Soportar tanto estructura simple como versionada
            arch_prefixes = [
                "models/cnn/",
                "models/cnn/v",       # Para estructura versionada
                "models/vgg16/",
                "models/vgg16/v",     # Para estructura versionada
                "models/efficientnet/",
                "models/efficientnet/v",  # Para estructura versionada
                "models/"  # Compatibilidad con modelos antiguos
            ]

            for prefix in arch_prefixes:
                # Para prefijos versionados, buscar en todas las versiones
                if prefix.endswith('/v'):
                    # Buscar versiones (v1/, v2/, etc.)
                    version_blobs = list(bucket.list_blobs(prefix=prefix))
                    version_prefixes = set()
                    for blob in version_blobs:
                        # Extraer el prefijo de versión (ej: models/efficientnet/v1/)
                        path_parts = blob.name.split('/')
                        if len(path_parts) >= 3 and path_parts[2].startswith('v'):
                            version_prefix = '/'.join(path_parts[:3]) + '/'
                            version_prefixes.add(version_prefix)

                    # Buscar en cada versión encontrada
                    for version_prefix in version_prefixes:
                        blobs = list(bucket.list_blobs(prefix=version_prefix))
                        for blob in blobs:
                            if not blob.name.endswith('/'):
                                if blob.name.endswith(('.keras', '.h5', '.weights.h5')):
                                    if not blob.name.endswith('_config.json'):
                                        min_size = 5 * 1024 * 1024 if blob.name.endswith('.weights.h5') else 1024 * 1024
                                        if blob.size and blob.size > min_size:
                                            all_blobs.append(blob)
                                            print(f"   📦 Encontrado (versionado): {blob.name} ({blob.size / (1024*1024):.1f}MB)")
                else:
                    # Búsqueda normal para estructura simple
                    blobs = list(bucket.list_blobs(prefix=prefix))

                    # Filtrar blobs que son archivos de modelo válidos
                    for blob in blobs:
                        # Solo archivos (no directorios vacíos)
                        if not blob.name.endswith('/'):
                            # Solo archivos de modelo
                            if blob.name.endswith(('.keras', '.h5', '.weights.h5')):
                                # Excluir archivos de configuración
                                if not blob.name.endswith('_config.json'):
                                    # Filtrar por tamaño mínimo para evitar archivos corruptos
                                    min_size = 5 * 1024 * 1024 if blob.name.endswith('.weights.h5') else 1024 * 1024  # 5MB para weights, 1MB para otros
                                    if blob.size and blob.size > min_size:
                                        # Para modelos en directorio base, solo archivos directos
                                        if prefix != "models/" or "/" not in blob.name[7:]:
                                            all_blobs.append(blob)
                                            print(f"   📦 Encontrado: {blob.name} ({blob.size / (1024*1024):.1f}MB)")
                                    else:
                                        print(f"   ⚠️  Archivo muy pequeño (posiblemente corrupto): {blob.name} ({blob.size / (1024*1024):.1f}MB)")

            print(f"📊 Total de modelos válidos encontrados en GCS: {len(all_blobs)}")

            if not all_blobs:
                print(f"{Fore.RED}❌ No se encontraron modelos válidos en GCS bucket {BUCKET_NAME}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}🔄 Intentando fallback a almacenamiento local...{Style.RESET_ALL}")
                return load_from_local()

            # Priorizar modelos de la arquitectura configurada
            from coffeedd.params import MODEL_ARCHITECTURE
            preferred_arch = MODEL_ARCHITECTURE.lower()

            # Filtrar modelos de la arquitectura preferida
            preferred_blobs = []
            for blob in all_blobs:
                # Detectar arquitectura desde el nombre del archivo o ruta
                detected_arch = detect_model_architecture(blob.name)
                if detected_arch == preferred_arch:
                    preferred_blobs.append(blob)

            if preferred_blobs:
                print(f"🎯 Encontrados {len(preferred_blobs)} modelos de arquitectura preferida ({preferred_arch})")
                # Priorizar .weights.h5 dentro de la arquitectura preferida
                weights_blobs = [blob for blob in preferred_blobs if blob.name.endswith('.weights.h5')]
                if weights_blobs:
                    latest_blob = max(weights_blobs, key=lambda x: x.updated)
                    print(f"✅ Usando .weights.h5 de {preferred_arch}: {latest_blob.name}")
                else:
                    latest_blob = max(preferred_blobs, key=lambda x: x.updated)
                    print(f"✅ Usando modelo de {preferred_arch}: {latest_blob.name}")
            else:
                print(f"⚠️  No se encontraron modelos válidos de {preferred_arch}, usando el más reciente disponible")
                # Fallback: usar cualquier modelo válido
                weights_blobs = [blob for blob in all_blobs if blob.name.endswith('.weights.h5')]
                if weights_blobs:
                    latest_blob = max(weights_blobs, key=lambda x: x.updated)
                    print(f"🔄 Priorizando archivo .weights.h5: {latest_blob.name}")
                else:
                    latest_blob = max(all_blobs, key=lambda x: x.updated)
                    print(f"� Usando modelo más reciente: {latest_blob.name}")

            print(f"📦 Modelo seleccionado: {latest_blob.name}")
            print(f"📅 Última actualización: {latest_blob.updated}")
            print(f"📏 Tamaño: {latest_blob.size / (1024*1024):.2f} MB")

            # Detectar arquitectura desde la ruta del blob
            blob_path_parts = latest_blob.name.split('/')
            detected_architecture = None
            if len(blob_path_parts) >= 2 and blob_path_parts[0] == 'models':
                potential_arch = blob_path_parts[1]
                if potential_arch in ['cnn', 'vgg16', 'efficientnet']:
                    detected_architecture = potential_arch
                    print(f"🔍 Arquitectura detectada desde GCS: {detected_architecture}")

            # Crear directorio local manteniendo estructura de arquitectura
            if detected_architecture:
                local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", detected_architecture)
            else:
                local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
            os.makedirs(local_model_directory, exist_ok=True)

            # Descargar archivo principal
            model_filename = os.path.basename(latest_blob.name)
            latest_model_path_to_save = os.path.join(local_model_directory, model_filename)

            print(f"📥 Descargando a: {latest_model_path_to_save}")
            latest_blob.download_to_filename(latest_model_path_to_save)

            # Si es archivo de pesos, buscar también el archivo de configuración
            config_path = None
            if latest_blob.name.endswith('.weights.h5'):
                config_blob_name = latest_blob.name.replace('.weights.h5', '_config.json')

                try:
                    config_blob = bucket.blob(config_blob_name)
                    if config_blob.exists():
                        config_filename = os.path.basename(config_blob_name)
                        config_path = os.path.join(local_model_directory, config_filename)
                        config_blob.download_to_filename(config_path)
                        print(f"📥 Configuración descargada: {config_path}")
                    else:
                        print(f"{Fore.YELLOW}⚠️  Archivo de configuración no encontrado: {config_blob_name}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠️  Error descargando configuración: {e}{Style.RESET_ALL}")

            # Ahora cargar el modelo usando la misma lógica que load_from_local
            try:
                print(f"🔄 Cargando modelo descargado...")

                # Si es archivo de pesos (.weights.h5), reconstruir el modelo
                if latest_model_path_to_save.endswith('.weights.h5'):
                    print(Fore.BLUE + "🔧 Detectado archivo de pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                    # Leer la configuración
                    model_info = {}
                    if config_path and os.path.exists(config_path):
                        import json
                        with open(config_path, 'r') as f:
                            model_info = json.load(f)
                        print(f"📄 Configuración cargada: {model_info}")
                        useefficientnet = model_info.get('model_type') == 'EfficientNet'
                    else:
                        # Detectar por nombre del archivo
                        useefficientnet = 'EfficientNet' in model_filename or 'efficientnet' in model_filename.lower()
                        print(f"{Fore.YELLOW}⚠️  No hay configuración, detectando por nombre: {'EfficientNet' if useefficientnet else 'CNN'}{Style.RESET_ALL}")

                    # Lista de estrategias de reconstrucción para probar
                    reconstruction_strategies = []

                    if useefficientnet:
                        print("🏗️  Reconstruyendo EfficientNetB0...")
                        # Estrategias para EfficientNet (en orden de probabilidad)
                        reconstruction_strategies = [
                            ("sin fine-tuning", "standard"),
                            ("con fine-tuning", "finetuning"),
                        ]
                    else:
                        print("🏗️  Reconstruyendo CNN simple...")
                        # Estrategias para CNN simple
                        reconstruction_strategies = [
                            ("CNN estándar", "standard"),
                        ]

                    # Probar cada estrategia hasta encontrar una compatible
                    model_loaded = False
                    for strategy_name, strategy_type in reconstruction_strategies:
                        try:
                            print(f"🔍 Intentando estrategia: {strategy_name}...")

                            if useefficientnet:
                                if strategy_type == "finetuning":
                                    # Estrategia con fine-tuning
                                    model, base_model = build_efficientnet_model()

                                    # Aplicar configuración de fine-tuning
                                    base_model.trainable = True
                                    fine_tune_at = len(base_model.layers) - 15
                                    for layer in base_model.layers[:fine_tune_at]:
                                        layer.trainable = False

                                    print(f"🔧 Fine-tuning configurado: {fine_tune_at} capas congeladas, {15} entrenables")

                                    # Compilar con learning rate de fine-tuning
                                    model.compile(
                                        optimizer=keras.optimizers.Adam(learning_rate=0.001 / 10),
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy']
                                    )
                                else:
                                    # Estrategia estándar sin fine-tuning
                                    model, _ = build_efficientnet_model()

                                    # Compilar con métricas básicas
                                    model.compile(
                                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy']
                                    )
                            else:
                                # CNN simple
                                model = build_simple_cnn_model()

                                # Compilar con métricas básicas
                                model.compile(
                                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )

                            # Intentar cargar pesos
                            print(f"📊 Modelo reconstruido tiene {len(model.layers)} capas")

                            # Debug: Mostrar información de los pesos guardados
                            try:
                                import h5py
                                with h5py.File(latest_model_path_to_save, 'r') as f:
                                    print(f"📦 Archivo de pesos contiene información para {len(f.keys())} grupos")
                            except Exception:
                                print(f"📦 Analizando archivo de pesos...")

                            model.load_weights(latest_model_path_to_save)
                            print(Fore.GREEN + f"✅ Pesos cargados exitosamente ({strategy_name})" + Style.RESET_ALL)
                            model_loaded = True
                            break

                        except Exception as e:
                            error_msg = str(e)
                            if "Layer count mismatch" in error_msg:
                                print(f"❌ {strategy_name}: Incompatibilidad de capas - {error_msg}")
                            elif "axes don't match array" in error_msg:
                                print(f"❌ {strategy_name}: Incompatibilidad de dimensiones - {error_msg}")
                            else:
                                print(f"❌ {strategy_name}: Error - {error_msg}")
                            continue

                    if not model_loaded:
                        print(Fore.RED + "❌ No se pudo reconstruir el modelo con ninguna estrategia" + Style.RESET_ALL)
                        print(Fore.YELLOW + "💡 Información del archivo:" + Style.RESET_ALL)
                        print(f"   📁 Archivo: {latest_model_path_to_save}")
                        if model_info:
                            print(f"   📄 Config: {model_info}")
                        raise ValueError("No se pudo reconstruir el modelo compatible con los pesos guardados")

                else:
                    # Intentar cargar modelo completo (.keras o .h5)
                    try:
                        model = keras.models.load_model(
                            latest_model_path_to_save,
                            custom_objects={'DiseaseRecallMetric': DiseaseRecallMetric}
                        )

                        useefficientnet = 'EfficientNet' in model_filename or 'efficientnet' in model_filename.lower()

                    except ValueError as e:
                        if "No model config found" in str(e):
                            # Es un archivo de solo pesos con extensión .h5/.keras
                            print(Fore.YELLOW + "⚠️  Archivo contiene solo pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                            # Detectar tipo por nombre del archivo
                            useefficientnet = 'EfficientNet' in model_filename or 'efficientnet' in model_filename.lower()

                            print(f"🔍 Detectado tipo: {'EfficientNet' if useefficientnet else 'CNN simple'} basado en nombre: {model_filename}")

                            # Usar la misma lógica de múltiples estrategias que para .weights.h5
                            reconstruction_strategies = []

                            if useefficientnet:
                                print("🏗️  Reconstruyendo EfficientNetB0...")
                                reconstruction_strategies = [
                                    ("sin fine-tuning", "standard"),
                                    ("con fine-tuning", "finetuning"),
                                ]
                            else:
                                print("🏗️  Reconstruyendo CNN simple...")
                                reconstruction_strategies = [
                                    ("CNN estándar", "standard"),
                                ]

                            # Probar cada estrategia hasta encontrar una compatible
                            model_loaded = False
                            for strategy_name, strategy_type in reconstruction_strategies:
                                try:
                                    print(f"🔍 Intentando estrategia: {strategy_name}...")

                                    if useefficientnet:
                                        if strategy_type == "finetuning":
                                            # Estrategia con fine-tuning
                                            model, base_model = build_efficientnet_model()

                                            # Aplicar configuración de fine-tuning
                                            base_model.trainable = True
                                            fine_tune_at = len(base_model.layers) - 15
                                            for layer in base_model.layers[:fine_tune_at]:
                                                layer.trainable = False

                                            print(f"🔧 Fine-tuning configurado: {fine_tune_at} capas congeladas, {15} entrenables")

                                            # Compilar con learning rate de fine-tuning
                                            model.compile(
                                                optimizer=keras.optimizers.Adam(learning_rate=0.001 / 10),
                                                loss='categorical_crossentropy',
                                                metrics=['accuracy']
                                            )
                                        else:
                                            # Estrategia estándar sin fine-tuning
                                            model, _ = build_efficientnet_model()

                                            # Compilar con métricas básicas
                                            model.compile(
                                                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                                loss='categorical_crossentropy',
                                                metrics=['accuracy']
                                            )
                                    else:
                                        # CNN simple
                                        model = build_simple_cnn_model()

                                        # Compilar con métricas básicas
                                        model.compile(
                                            optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                            loss='categorical_crossentropy',
                                            metrics=['accuracy']
                                        )

                                    # Intentar cargar pesos
                                    print(f"📊 Modelo reconstruido tiene {len(model.layers)} capas")

                                    # Debug: Mostrar información de los pesos guardados
                                    try:
                                        import h5py
                                        with h5py.File(latest_model_path_to_save, 'r') as f:
                                            print(f"📦 Archivo de pesos contiene información para {len(f.keys())} grupos")
                                    except Exception:
                                        print(f"📦 Analizando archivo de pesos...")

                                    model.load_weights(latest_model_path_to_save)
                                    print(Fore.GREEN + f"✅ Pesos cargados exitosamente ({strategy_name})" + Style.RESET_ALL)
                                    model_loaded = True
                                    break

                                except Exception as load_error:
                                    error_msg = str(load_error)
                                    if "Layer count mismatch" in error_msg:
                                        print(f"❌ {strategy_name}: Incompatibilidad de capas - {error_msg}")
                                    elif "axes don't match array" in error_msg:
                                        print(f"❌ {strategy_name}: Incompatibilidad de dimensiones - {error_msg}")
                                    else:
                                        print(f"❌ {strategy_name}: Error - {error_msg}")
                                    continue

                            if not model_loaded:
                                print(Fore.RED + "❌ No se pudo reconstruir el modelo con ninguna estrategia" + Style.RESET_ALL)
                                print(Fore.YELLOW + "💡 Información del archivo:" + Style.RESET_ALL)
                                print(f"   📁 Archivo: {latest_model_path_to_save}")
                                print(f"   🏷️  Nombre: {model_filename}")
                                print(f"   🔍 Tipo detectado: {'EfficientNet' if useefficientnet else 'CNN'}")
                                raise ValueError("No se pudo reconstruir el modelo compatible con los pesos guardados")

                        else:
                            raise

                # Recompilar con todas las métricas si se solicita
                if compile_with_metrics:
                    print(Fore.BLUE + "🔧 Recompilando con métricas completas..." + Style.RESET_ALL)
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=[
                            'accuracy',
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.Precision(name='precision'),
                            DiseaseRecallMetric(),
                            keras.metrics.AUC(name='auc')
                        ]
                    )

                print(Fore.GREEN + "✅ Modelo cargado exitosamente desde GCS" + Style.RESET_ALL)

                # Detectar arquitectura correctamente
                final_architecture = detect_model_architecture(model.layers)
                arch_display_names = {
                    'efficientnet': 'EfficientNetB0',
                    'vgg16': 'VGG16',
                    'cnn': 'CNN simple'
                }
                display_name = arch_display_names.get(final_architecture, final_architecture)
                print(f"🏷️  Tipo: {display_name}")

                print(f"📁 Archivo local: {latest_model_path_to_save}")

                return model

            except Exception as e:
                print(Fore.RED + f"❌ Error al cargar modelo descargado: {e}" + Style.RESET_ALL)
                import traceback
                traceback.print_exc()

                print(f"{Fore.YELLOW}🔄 Intentando fallback a almacenamiento local...{Style.RESET_ALL}")
                return load_from_local()

        except Exception as e:
            print(Fore.RED + f"❌ Error al conectar con GCS: {e}" + Style.RESET_ALL)
            print(f"{Fore.YELLOW}🔄 Intentando fallback a almacenamiento local...{Style.RESET_ALL}")
            return load_from_local()

    elif MODEL_TARGET == "mlflow":
        print(Fore.MAGENTA + f"\n📥 Cargando modelo desde MLflow (stage: {stage})..." + Style.RESET_ALL)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            # Intentar obtener el modelo del stage especificado
            model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])

            if not model_versions:
                print(Fore.YELLOW + f"⚠️  No se encontró modelo en stage '{stage}'" + Style.RESET_ALL)
                print(Fore.BLUE + "🔍 Intentando cargar la última versión disponible..." + Style.RESET_ALL)

                # Intentar obtener cualquier versión del modelo
                all_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
                if not all_versions:
                    print(Fore.RED + f"❌ No se encontró ningún modelo '{MLFLOW_MODEL_NAME}' en MLflow" + Style.RESET_ALL)
                    print(Fore.YELLOW + "🔄 Intentando fallback a almacenamiento local..." + Style.RESET_ALL)
                    return load_from_local()

                # Usar la versión más reciente
                model_versions = [max(all_versions, key=lambda x: int(x.version))]
                print(f"📦 Usando versión {model_versions[0].version}")

            model_uri = model_versions[0].source
            model_version = model_versions[0].version
            run_id = model_versions[0].run_id

            print(f"📦 Modelo: {MLFLOW_MODEL_NAME} v{model_version}")
            print(f"🔗 URI: {model_uri}")

            # Obtener info del run
            run = client.get_run(run_id)

            # Verificar si es solo pesos o modelo completo
            storage_format = run.data.tags.get('storage_format', 'complete_model')

            if storage_format == 'weights_only':
                print(Fore.BLUE + "🔧 Detectado formato de pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                # Descargar artifacts
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    artifacts_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path="model",
                        dst_path=tmp_dir
                    )

                    # Cargar config
                    import json
                    config_path = os.path.join(artifacts_path, 'model_config.json')
                    with open(config_path, 'r') as f:
                        model_info = json.load(f)

                    useefficientnet = model_info['model_type'] == 'EfficientNet'

                    # Reconstruir arquitectura
                    if useefficientnet:
                        print("🏗️  Reconstruyendo EfficientNetB0...")
                        model, _ = build_efficientnet_model()
                    else:
                        print("🏗️  Reconstruyendo CNN simple...")
                        model = build_simple_cnn_model()

                    # Compilar y cargar pesos
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy', 'recall', 'precision', 'auc']
                    )

                    weights_path = os.path.join(artifacts_path, 'model.weights.h5')
                    model.load_weights(weights_path)
                    print(Fore.GREEN + "✅ Pesos cargados exitosamente" + Style.RESET_ALL)
            else:
                # Cargar modelo completo
                try:
                    model = mlflow.tensorflow.load_model(model_uri)
                    useefficientnet = 'efficientnet' in MLFLOW_MODEL_NAME.lower()

                except Exception as e:
                    print(Fore.YELLOW + f"⚠️  Intentando método alternativo..." + Style.RESET_ALL)

                    # Descargar y cargar con keras
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_model_path = mlflow.artifacts.download_artifacts(
                            artifact_uri=f"{model_uri}/model",
                            dst_path=tmp_dir
                        )

                        model = keras.models.load_model(
                            local_model_path,
                            custom_objects={'DiseaseRecallMetric': DiseaseRecallMetric}
                        )
                        useefficientnet = 'efficientnet' in MLFLOW_MODEL_NAME.lower()

            # Obtener tipo desde parámetros si está disponible
            if 'useefficientnet' in run.data.params:
                useefficientnet = run.data.params['useefficientnet'].lower() == 'true'

            # Recompilar con todas las métricas si se solicita
            if compile_with_metrics:
                print(Fore.BLUE + "🔧 Recompilando con métricas completas..." + Style.RESET_ALL)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=[
                        'accuracy',
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.Precision(name='precision'),
                        DiseaseRecallMetric(),
                        keras.metrics.AUC(name='auc')
                    ]
                )

            print(Fore.GREEN + "✅ Modelo cargado exitosamente desde MLflow" + Style.RESET_ALL)

            # Detectar arquitectura correctamente
            final_architecture = detect_model_architecture(model.layers)
            arch_display_names = {
                'efficientnet': 'EfficientNetB0',
                'vgg16': 'VGG16',
                'cnn': 'CNN simple'
            }
            display_name = arch_display_names.get(final_architecture, final_architecture)
            print(f"🏷️  Tipo: {display_name}")

            print(f"📊 Run ID: {run_id}")

            return model

        except Exception as e:
            print(Fore.RED + f"❌ Error al cargar modelo desde MLflow: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "🔄 Intentando fallback a almacenamiento local..." + Style.RESET_ALL)
            import traceback
            traceback.print_exc()

            # FALLBACK: Intentar cargar desde local
            return load_from_local()

    else:
        print(Fore.RED + f"❌ MODEL_TARGET no válido: '{MODEL_TARGET}'" + Style.RESET_ALL)
        print(Fore.YELLOW + "💡 Valores permitidos: 'local' o 'mlflow'" + Style.RESET_ALL)
        return None

def save_results(params: dict, metrics: dict):
    """Guarda los parámetros y métricas del modelo ya sea en MLflow o localmente.
    Args:
        params (dict): Diccionario con los parámetros del modelo.
        metrics (dict): Diccionario con las métricas del modelo.
    """
    # Debug: verificar estado de MLflow
    import mlflow
    active_run = mlflow.active_run()

    print(f"\n🔍 DEBUG - save_results:")
    print(f"   MODEL_TARGET: {MODEL_TARGET}")
    print(f"   MLflow active run: {active_run is not None}")
    if active_run:
        print(f"   Run ID: {active_run.info.run_id}")
    print(f"   Params: {len(params) if params else 0} items")
    print(f"   Metrics: {len(metrics) if metrics else 0} items")

    # Intentar loggear a MLflow si hay contexto activo o MODEL_TARGET es mlflow
    mlflow_logged = False
    if active_run or MODEL_TARGET == "mlflow":
        try:
            if params is not None:
                mlflow.log_params(params)
                print(f"✅ Logged {len(params)} params to MLflow")
            if metrics is not None:
                mlflow.log_metrics(metrics)
                print(f"✅ Logged {len(metrics)} metrics to MLflow")
            mlflow_logged = True
            print(Fore.GREEN + "\n✅ Resultados guardados en MLflow." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️  Error logging to MLflow: {e}" + Style.RESET_ALL)
            mlflow_logged = False
    else:
        print(Fore.YELLOW + "⚠️  No active MLflow run and MODEL_TARGET != 'mlflow'" + Style.RESET_ALL)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    #Guardar params localmente
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
                pickle.dump(params, file)

    #Guardar métricas localmente
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
                pickle.dump(metrics, file)

    print(Fore.GREEN + f"\n✅ Resultados guardados localmente ({'y MLflow' if mlflow_logged else 'solamente'})." + Style.RESET_ALL)

def save_model(model: keras.Model = None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Detectar tipo de modelo usando la nueva función
    architecture = detect_model_architecture(model.layers)

    # Mapear a nombres de archivos
    arch_map = {
        'efficientnet': 'EfficientNet',
        'vgg16': 'VGG16',
        'cnn': 'CNN'
    }
    model_type = arch_map.get(architecture, 'CNN')

    # Crear estructura de carpetas por arquitectura
    models_base_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    architecture_dir = os.path.join(models_base_dir, architecture)
    os.makedirs(architecture_dir, exist_ok=True)

    # Usar convención de nombres consistente
    model_filename = f"model_{model_type}_{timestamp}.keras"
    model_path = os.path.join(architecture_dir, model_filename)

    # Variables para tracking del guardado exitoso
    weights_saved = False
    saved_file_path = None

    try:
        model.save(model_path)
        print(Fore.GREEN + f"\n✅ Modelo guardado localmente en: {model_path}" + Style.RESET_ALL)
        saved_file_path = model_path

    except TypeError as e:
        if "Unable to serialize" in str(e) or "EagerTensor" in str(e):
            print(Fore.YELLOW + "⚠️  Error de serialización detectado" + Style.RESET_ALL)
            print(Fore.BLUE + "🔄 Guardando en formato de pesos separados..." + Style.RESET_ALL)

            # Eliminar archivo .keras corrupto si se creó
            if os.path.exists(model_path) and os.path.getsize(model_path) == 0:
                os.remove(model_path)
                print(f"🗑️  Eliminado archivo .keras corrupto: {model_path}")

            # Guardar pesos con convención consistente
            weights_filename = f"model_{model_type}_{timestamp}.weights.h5"
            weights_path = os.path.join(architecture_dir, weights_filename)
            model.save_weights(weights_path)
            weights_saved = True
            saved_file_path = weights_path

            # Guardar configuración del modelo (arquitectura)
            config_filename = f"model_{model_type}_{timestamp}_config.json"
            config_path = os.path.join(architecture_dir, config_filename)
            import json

            # Guardar info básica sobre el modelo
            model_info = {
                'timestamp': timestamp,
                'input_shape': list(model.input_shape) if model.input_shape else None,
                'output_shape': list(model.output_shape) if model.output_shape else None,
                'num_layers': len(model.layers),
                'model_type': model_type,
                'architecture': architecture
            }

            with open(config_path, 'w') as f:
                json.dump(model_info, f, indent=2)

            print(Fore.GREEN + f"\n✅ Modelo guardado en formato de pesos:" + Style.RESET_ALL)
            print(f"   📦 Pesos: {weights_path}")
            print(f"   ⚙️  Config: {config_path}")
            print(Fore.BLUE + "ℹ️  Para cargar: usar load_model() que reconstruirá el modelo" + Style.RESET_ALL)
        else:
            raise e

    if MODEL_TARGET == "gcs":
        from google.cloud import storage

        # Solo subir si tenemos un archivo válido guardado
        if saved_file_path and os.path.exists(saved_file_path):
            # Usar la estructura de carpetas por arquitectura
            model_filename = os.path.basename(saved_file_path)
            gcs_model_path = f"models/{architecture}/{model_filename}"

            print(f"☁️  Guardando en GCS: {gcs_model_path}")

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(gcs_model_path)
            blob.upload_from_filename(saved_file_path)

            # Si es un archivo de pesos, también subir el archivo de configuración
            if saved_file_path.endswith('.weights.h5'):
                config_filename = model_filename.replace('.weights.h5', '_config.json')
                config_path = os.path.join(os.path.dirname(saved_file_path), config_filename)
                if os.path.exists(config_path):
                    gcs_config_path = f"models/{architecture}/{config_filename}"
                    config_blob = bucket.blob(gcs_config_path)
                    config_blob.upload_from_filename(config_path)
                    print(f"☁️  Configuración guardada en GCS: {gcs_config_path}")

            print(f"✅ Modelo guardado en GCS: {gcs_model_path}")
        else:
            print(f"❌ Error: No se pudo subir a GCS, archivo no válido: {saved_file_path}")

        return None

    if MODEL_TARGET == "mlflow":
        # Guardar en MLflow usando la nueva arquitectura
        save_model_to_mlflow(model=model, params=None, metrics=None, architecture=architecture)

    return None

def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ El modelo {MLFLOW_MODEL_NAME} ha sido movido de {current_stage} a {new_stage}.")

    return None

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        # Debug logging
        print(f"\n🚀 MLflow wrapper starting for {func.__name__}")

        # Solo terminar run si hay uno activo y no es el que queremos
        active_run = mlflow.active_run()
        if active_run:
            print(f"⚠️  Found active run {active_run.info.run_id}, ending it")
            mlflow.end_run()

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        print(f"🎯 Starting MLflow run for {func.__name__}")
        with mlflow.start_run() as run:
            print(f"🆔 MLflow Run ID: {run.info.run_id}")
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)
            print(f"✅ Function {func.__name__} completed, MLflow context still active")

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper

def save_model_to_mlflow(model, params, metrics, architecture):
    """Guarda modelo en MLflow, creando el registro si no existe."""

    import warnings
    import logging
    from mlflow.tracking import MlflowClient

    # Suprimir warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    client = MlflowClient()
    tmp_weights_path = None
    tmp_config_path = None

    # Mapear arquitectura a nombres legibles
    arch_display_names = {
        'efficientnet': 'EfficientNetB0',
        'vgg16': 'VGG16',
        'cnn': 'CNN'
    }
    display_name = arch_display_names.get(architecture, architecture.upper())

    # Verificar si el modelo registrado existe, si no, crearlo
    try:
        client.get_registered_model(MLFLOW_MODEL_NAME)
        print(f"✅ Modelo registrado '{MLFLOW_MODEL_NAME}' encontrado")
    except:
        print(Fore.YELLOW + f"⚠️  Modelo '{MLFLOW_MODEL_NAME}' no existe, creándolo..." + Style.RESET_ALL)
        try:
            client.create_registered_model(
                name=MLFLOW_MODEL_NAME,
                description=f"Coffee Disease Detection Model - {display_name}"
            )
            print(Fore.GREEN + f"✅ Modelo registrado '{MLFLOW_MODEL_NAME}' creado exitosamente" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Error al crear modelo registrado: {e}" + Style.RESET_ALL)

    try:
        with mlflow.start_run() as run:

            # Agregar metadata
            params = params or {}
            params['model_architecture'] = architecture
            params['model_display_name'] = display_name

            # Log params y metrics
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)

            print(f"🆔 Run ID: {run.info.run_id}")

            # Intentar guardar el modelo completo primero
            model_save_success = False

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                try:
                    # Intentar guardar en formato .keras
                    model.save(tmp.name)
                    model_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
                    tmp_path = tmp.name
                    model_save_success = True
                    print(f"📦 Tamaño del modelo: {model_size_mb:.2f} MB")

                except TypeError as e:
                    if "Unable to serialize" in str(e) or "EagerTensor" in str(e):
                        print(Fore.YELLOW + "⚠️  No se puede serializar modelo completo, usando formato de pesos..." + Style.RESET_ALL)
                        # Guardar pesos y config por separado
                        tmp_weights_path = tmp.name.replace('.keras', '.weights.h5')
                        model.save_weights(tmp_weights_path)

                        # Guardar config
                        import json
                        tmp_config_path = tmp.name.replace('.keras', '_config.json')
                        model_info = {
                            'input_shape': list(model.input_shape) if model.input_shape else None,
                            'output_shape': list(model.output_shape) if model.output_shape else None,
                            'num_layers': len(model.layers),
                            'model_type': display_name,
                            'architecture': architecture
                        }
                        with open(tmp_config_path, 'w') as f:
                            json.dump(model_info, f, indent=2)

                        model_size_mb = os.path.getsize(tmp_weights_path) / (1024 * 1024)
                        print(f"📦 Tamaño de pesos: {model_size_mb:.2f} MB")
                    else:
                        raise

            mlflow.log_param("model_size_mb", round(model_size_mb, 2))

            # Intentar subir a MLflow
            if model_size_mb < 50:  # Umbral de 50MB
                try:
                    print(Fore.BLUE + "💾 Guardando en MLflow..." + Style.RESET_ALL)

                    if model_save_success:
                        # Modelo completo
                        mlflow.tensorflow.log_model(
                            model=model,
                            artifact_path="model",
                            registered_model_name=MLFLOW_MODEL_NAME
                        )
                        mlflow.set_tag("storage_format", "complete_model")
                    else:
                        # Solo pesos y config
                        mlflow.log_artifact(tmp_weights_path, artifact_path="model")
                        mlflow.log_artifact(tmp_config_path, artifact_path="model")
                        mlflow.set_tag("storage_format", "weights_only")
                        mlflow.set_tag("requires_reconstruction", "true")

                    mlflow.set_tag("storage", "mlflow")
                    print(Fore.GREEN + "✅ Modelo guardado en MLflow" + Style.RESET_ALL)

                    # Limpiar temporales
                    if model_save_success and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    if tmp_weights_path and os.path.exists(tmp_weights_path):
                        os.unlink(tmp_weights_path)
                    if tmp_config_path and os.path.exists(tmp_config_path):
                        os.unlink(tmp_config_path)

                except Exception as e:
                    print(Fore.YELLOW + f"⚠️  Error al subir: {str(e)[:100]}" + Style.RESET_ALL)
                    raise

            else:
                print(Fore.YELLOW + f"⚠️  Modelo muy grande ({model_size_mb:.1f}MB)" + Style.RESET_ALL)
                raise ValueError("Model too large for MLflow")

    except Exception as e:
        # Fallback: guardar localmente
        print(Fore.YELLOW + f"💾 Guardando localmente como fallback... ({str(e)[:50]})" + Style.RESET_ALL)

        try:
            if not mlflow.active_run():
                mlflow.start_run()

            timestamp = int(time.time())

            # Intentar guardar modelo completo
            try:
                model_name = f"model_{display_name}_{timestamp}.keras"
                local_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                model.save(local_path)

            except TypeError:
                # Si falla, guardar pesos
                print(Fore.BLUE + "🔄 Guardando solo pesos localmente..." + Style.RESET_ALL)
                weights_name = f"model_{display_name}_{timestamp}.weights.h5"
                local_path = os.path.join(LOCAL_REGISTRY_PATH, "models", weights_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                model.save_weights(local_path)

                # Guardar config
                import json
                config_path = os.path.join(LOCAL_REGISTRY_PATH, "models",
                                          f"model_{display_name}_{timestamp}_config.json")
                model_info = {
                    'timestamp': timestamp,
                    'input_shape': list(model.input_shape) if model.input_shape else None,
                    'output_shape': list(model.output_shape) if model.output_shape else None,
                    'num_layers': len(model.layers),
                    'model_type': display_name,
                    'architecture': architecture
                }
                with open(config_path, 'w') as f:
                    json.dump(model_info, f, indent=2)

            mlflow.log_param("model_path", local_path)
            mlflow.log_param("storage", "local")
            mlflow.set_tag("storage", "local")
            mlflow.set_tag("error", str(e)[:200])

            print(Fore.GREEN + f"✅ Modelo guardado localmente: {local_path}" + Style.RESET_ALL)

            mlflow.end_run(status="FINISHED")

        except Exception as fallback_error:
            print(Fore.RED + f"❌ Error en fallback: {fallback_error}" + Style.RESET_ALL)
            mlflow.end_run(status="FAILED")
            raise
        finally:
            # Limpiar archivos temporales si aún existen
            for tmp_file in [tmp_weights_path, tmp_config_path]:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.unlink(tmp_file)
                    except:
                        pass

    return None

def load_specific_model(model_path: str) -> keras.Model:
    """
    Carga un modelo específico desde una ruta dada (para modelos en producción)

    Args:
        model_path: Ruta absoluta al archivo del modelo

    Returns:
        keras.Model: Modelo cargado o None si hay error
    """
    from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric

    if not os.path.exists(model_path):
        print(Fore.RED + f"❌ Archivo no encontrado: {model_path}" + Style.RESET_ALL)
        return None

    try:
        print(Fore.BLUE + f"📁 Cargando modelo específico: {os.path.basename(model_path)}" + Style.RESET_ALL)

        # Intentar cargar modelo completo primero
        try:
            model = keras.models.load_model(
                model_path,
                custom_objects={'DiseaseRecallMetric': DiseaseRecallMetric}
            )

            print(Fore.GREEN + "✅ Modelo completo cargado exitosamente" + Style.RESET_ALL)

        except Exception as e:
            print(Fore.YELLOW + f"⚠️  No se pudo cargar como modelo completo: {str(e)[:100]}" + Style.RESET_ALL)
            print(Fore.YELLOW + "🔄 Intentando carga mediante load_model() genérico..." + Style.RESET_ALL)

            # Fallback: usar la función load_model existente que ya maneja reconstrucción
            model = load_model()

            if model is None:
                print(Fore.RED + "❌ No se pudo cargar el modelo con ningún método" + Style.RESET_ALL)
                return None

        # Recompilar con métricas completas
        print(Fore.BLUE + "🔧 Recompilando con métricas completas..." + Style.RESET_ALL)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Recall(name='recall'),
                keras.metrics.Precision(name='precision'),
                DiseaseRecallMetric(),
                keras.metrics.AUC(name='auc')
            ]
        )

        print(Fore.GREEN + "✅ Modelo específico cargado exitosamente" + Style.RESET_ALL)
        return model

    except Exception as e:
        print(Fore.RED + f"❌ Error al cargar modelo específico: {e}" + Style.RESET_ALL)
        return None
