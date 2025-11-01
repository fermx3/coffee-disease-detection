import os
import glob
import time
import shutil
import tempfile
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from colorama import Fore, Style
from typing import Tuple
from google.cloud import storage

import keras
from coffeedd.params import MODEL_TARGET, LOCAL_REGISTRY_PATH, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, BUCKET_NAME

def load_model(stage="Production", compile_with_metrics=True) -> Tuple[keras.Model, bool]:
    from coffeedd.ml_logic.custom_metrics import DiseaseRecallMetric
    from coffeedd.ml_logic.model import build_simple_cnn_model, build_efficientnet_model

    # Función auxiliar para cargar desde local
    def load_from_local():
        print(Fore.MAGENTA + "\n📥 Cargando modelo desde almacenamiento local..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

        if not os.path.exists(local_model_directory):
            print(Fore.RED + f"❌ Directorio no existe: {local_model_directory}" + Style.RESET_ALL)
            return None, False

        # Buscar archivos .keras, .weights.h5 y directorios SavedModel
        all_items = os.listdir(local_model_directory)

        keras_files = [
            os.path.join(local_model_directory, item)
            for item in all_items
            if item.endswith('.keras')
        ]

        weights_files = [
            os.path.join(local_model_directory, item)
            for item in all_items
            if item.endswith('.weights.h5')
        ]

        savedmodel_dirs = [
            os.path.join(local_model_directory, item)
            for item in all_items
            if os.path.isdir(os.path.join(local_model_directory, item))
            and not item.startswith('.')
        ]

        local_model_paths = keras_files + weights_files + savedmodel_dirs

        if not local_model_paths:
            print(Fore.RED + "❌ No se encontraron modelos en el almacenamiento local." + Style.RESET_ALL)
            return None, False

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        try:
            print(f"📂 Cargando: {os.path.basename(most_recent_model_path_on_disk)}")

            # Si es archivo de pesos (.weights.h5), reconstruir el modelo
            if most_recent_model_path_on_disk.endswith('.weights.h5'):
                print(Fore.BLUE + "🔧 Detectado archivo de pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                # Leer la configuración
                timestamp = os.path.basename(most_recent_model_path_on_disk).replace('.weights.h5', '')
                config_path = os.path.join(local_model_directory, f"{timestamp}_config.json")

                import json
                with open(config_path, 'r') as f:
                    model_info = json.load(f)

                useefficientnet = model_info['model_type'] == 'EfficientNet'

                # Reconstruir la arquitectura
                if useefficientnet:
                    print("🏗️  Reconstruyendo EfficientNetB0...")
                    model, _ = build_efficientnet_model()
                else:
                    print("🏗️  Reconstruyendo CNN simple...")
                    model = build_simple_cnn_model()

                # Compilar el modelo (necesario antes de cargar pesos)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'recall', 'precision', 'auc']
                )

                # Cargar los pesos
                model.load_weights(most_recent_model_path_on_disk)
                print(Fore.GREEN + "✅ Pesos cargados exitosamente" + Style.RESET_ALL)

            else:
                # Intentar cargar modelo completo (.keras o SavedModel)
                try:
                    model = keras.models.load_model(
                        most_recent_model_path_on_disk,
                        custom_objects={'DiseaseRecallMetric': DiseaseRecallMetric}
                    )

                    useefficientnet = 'EfficientNet' in os.path.basename(most_recent_model_path_on_disk)

                except ValueError as e:
                    if "No model config found" in str(e):
                        # Es un archivo de solo pesos con extensión .keras
                        print(Fore.YELLOW + "⚠️  Archivo contiene solo pesos, reconstruyendo modelo..." + Style.RESET_ALL)

                        # Detectar tipo por nombre del archivo
                        useefficientnet = 'EfficientNet' in os.path.basename(most_recent_model_path_on_disk)

                        # Reconstruir arquitectura
                        if useefficientnet:
                            print("🏗️  Reconstruyendo EfficientNetB0...")
                            model, _ = build_efficientnet_model()
                        else:
                            print("🏗️  Reconstruyendo CNN simple...")
                            model = build_simple_cnn_model()

                        # Compilar antes de cargar pesos
                        model.compile(
                            optimizer=keras.optimizers.Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy', 'recall', 'precision', 'auc']
                        )

                        # Cargar pesos
                        model.load_weights(most_recent_model_path_on_disk)
                        print(Fore.GREEN + "✅ Pesos cargados exitosamente" + Style.RESET_ALL)
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
            print(f"🏷️  Tipo: {'EfficientNetB0' if useefficientnet else 'CNN simple'}")

            return model, useefficientnet

        except Exception as e:
            print(Fore.RED + f"❌ Error al cargar modelo local: {e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc()
            return None, False

    # ============================================================
    # LÓGICA PRINCIPAL CON FALLBACK
    # ============================================================

    if MODEL_TARGET == "local":
        return load_from_local()

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + "\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None, False

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
            print(f"🏷️  Tipo: {'EfficientNetB0' if useefficientnet else 'CNN simple'}")
            print(f"📊 Run ID: {run_id}")

            return model, useefficientnet

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
        return None, False

def save_results(params: dict, metrics: dict):
    """Guarda los parámetros y métricas del modelo ya sea en MLflow o localmente.
    Args:
        params (dict): Diccionario con los parámetros del modelo.
        metrics (dict): Diccionario con las métricas del modelo.
    """
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print(Fore.GREEN + "\n✅ Resultados guardados en MLflow." + Style.RESET_ALL)

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

    print(Fore.GREEN + "\n✅ Resultados guardados localmente." + Style.RESET_ALL)

def save_model(model: keras.Model = None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")

    try:
        model.save(model_path)
        print(Fore.GREEN + f"\n✅ Modelo guardado localmente en: {model_path}" + Style.RESET_ALL)

    except TypeError as e:
        if "Unable to serialize" in str(e) or "EagerTensor" in str(e):
            print(Fore.YELLOW + "⚠️  Error de serialización detectado" + Style.RESET_ALL)
            print(Fore.BLUE + "🔄 Guardando en formato de pesos separados..." + Style.RESET_ALL)

            # Guardar pesos
            weights_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.weights.h5")
            model.save_weights(weights_path)

            # Guardar configuración del modelo (arquitectura)
            config_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}_config.json")
            import json

            # Guardar info básica sobre el modelo
            model_info = {
                'timestamp': timestamp,
                'input_shape': list(model.input_shape) if model.input_shape else None,
                'output_shape': list(model.output_shape) if model.output_shape else None,
                'num_layers': len(model.layers),
                'model_type': 'EfficientNet' if any('efficientnet' in layer.name.lower() for layer in model.layers) else 'CNN'
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
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Modelo guardado en GCS.")

        return None

    if MODEL_TARGET == "mlflow":
        # Detectar tipo de modelo
        useefficientnet = any('efficientnet' in layer.name.lower() for layer in model.layers)
        save_model_to_mlflow(model=model, params=None, metrics=None, useefficientnet=useefficientnet)

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
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper

def save_model_to_mlflow(model, params, metrics, useefficientnet):
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

    # Verificar si el modelo registrado existe, si no, crearlo
    try:
        client.get_registered_model(MLFLOW_MODEL_NAME)
        print(f"✅ Modelo registrado '{MLFLOW_MODEL_NAME}' encontrado")
    except:
        print(Fore.YELLOW + f"⚠️  Modelo '{MLFLOW_MODEL_NAME}' no existe, creándolo..." + Style.RESET_ALL)
        try:
            client.create_registered_model(
                name=MLFLOW_MODEL_NAME,
                description=f"Coffee Disease Detection Model - {'EfficientNetB0' if useefficientnet else 'CNN'}"
            )
            print(Fore.GREEN + f"✅ Modelo registrado '{MLFLOW_MODEL_NAME}' creado exitosamente" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"❌ Error al crear modelo registrado: {e}" + Style.RESET_ALL)

    try:
        with mlflow.start_run() as run:

            # Agregar metadata
            params = params or {}
            params['useefficientnet'] = useefficientnet
            params['model_architecture'] = 'EfficientNetB0' if useefficientnet else 'CNN'

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
                            'model_type': 'EfficientNet' if useefficientnet else 'CNN'
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
                model_name = f"model_{'EfficientNet' if useefficientnet else 'CNN'}_{timestamp}.keras"
                local_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                model.save(local_path)

            except TypeError:
                # Si falla, guardar pesos
                print(Fore.BLUE + "🔄 Guardando solo pesos localmente..." + Style.RESET_ALL)
                weights_name = f"model_{'EfficientNet' if useefficientnet else 'CNN'}_{timestamp}.weights.h5"
                local_path = os.path.join(LOCAL_REGISTRY_PATH, "models", weights_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                model.save_weights(local_path)

                # Guardar config
                import json
                config_path = os.path.join(LOCAL_REGISTRY_PATH, "models",
                                          f"model_{'EfficientNet' if useefficientnet else 'CNN'}_{timestamp}_config.json")
                model_info = {
                    'timestamp': timestamp,
                    'input_shape': list(model.input_shape) if model.input_shape else None,
                    'output_shape': list(model.output_shape) if model.output_shape else None,
                    'num_layers': len(model.layers),
                    'model_type': 'EfficientNet' if useefficientnet else 'CNN'
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
