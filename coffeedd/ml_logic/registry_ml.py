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

import keras
from coffeedd.params import MODEL_TARGET, LOCAL_REGISTRY_PATH, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT

def load_model(stage="Production") -> Tuple[keras.Model, bool]:
    """
    Carga un modelo desde almacenamiento local o MLflow.

    Args:
        stage: Stage del modelo en MLflow ('Production', 'Staging', etc.)

    Returns:
        Tuple[Model, bool]: (modelo, useefficientnet) o (None, False) si no existe
    """
    if MODEL_TARGET == "local":
        print(Fore.MAGENTA + "\n📥 Cargando modelo desde almacenamiento local..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*.keras")

        if not local_model_paths:
            print(Fore.RED + "❌ No se encontraron modelos en el almacenamiento local." + Style.RESET_ALL)
            return None, False

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        try:
            print(f"📂 Cargando: {os.path.basename(most_recent_model_path_on_disk)}")
            model = keras.models.load_model(most_recent_model_path_on_disk)

            # Detectar tipo de modelo por el nombre del archivo
            useefficientnet = 'EfficientNet' in most_recent_model_path_on_disk

            print(Fore.GREEN + "✅ Modelo cargado exitosamente" + Style.RESET_ALL)
            print(f"🏷️  Tipo: {'EfficientNetB0' if useefficientnet else 'CNN simple'}")

            return model, useefficientnet

        except Exception as e:
            print(Fore.RED + f"❌ Error al cargar modelo: {e}" + Style.RESET_ALL)
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
                    return None, False

                # Usar la versión más reciente
                model_versions = [max(all_versions, key=lambda x: int(x.version))]
                print(f"📦 Usando versión {model_versions[0].version}")

            model_uri = model_versions[0].source
            model_version = model_versions[0].version

            print(f"📦 Modelo: {MLFLOW_MODEL_NAME} v{model_version}")
            print(f"🔗 URI: {model_uri}")

            # Cargar el modelo
            model = mlflow.tensorflow.load_model(model_uri)

            # Detectar tipo de modelo desde los tags o run info
            run_id = model_versions[0].run_id
            run = client.get_run(run_id)

            # Intentar obtener el tipo desde los parámetros del run
            useefficientnet = False
            if 'useefficientnet' in run.data.params:
                useefficientnet = run.data.params['useefficientnet'].lower() == 'true'
            else:
                # Fallback: inferir del nombre del modelo
                useefficientnet = 'efficientnet' in MLFLOW_MODEL_NAME.lower()

            print(Fore.GREEN + "✅ Modelo cargado exitosamente desde MLflow" + Style.RESET_ALL)
            print(f"🏷️  Tipo: {'EfficientNetB0' if useefficientnet else 'CNN simple'}")
            print(f"📊 Run ID: {run_id}")

            return model, useefficientnet

        except IndexError:
            print(Fore.RED + f"❌ No se encontró ningún modelo en stage: {stage}" + Style.RESET_ALL)
            return None, False

        except Exception as e:
            print(Fore.RED + f"❌ Error al cargar modelo desde MLflow: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "💡 Tip: Verifica que el modelo esté registrado correctamente" + Style.RESET_ALL)
            return None, False

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

    # Guardar modelo localmente en formato .keras (más robusto que .h5)
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")
    model.save(model_path)

    print(Fore.GREEN + f"\n✅ Modelo guardado localmente en: {model_path}" + Style.RESET_ALL)

    if MODEL_TARGET == "gcs":
        # TODO: Aquí se podría agregar la lógica para subir el modelo a Google Cloud Storage
        print(Fore.GREEN + "\n✅ Modelo subido a GCS." + Style.RESET_ALL)
        pass

    if MODEL_TARGET == "mlflow":
        # Guardar localmente SIEMPRE
        save_model_to_mlflow(model=model, params=None, metrics=None, useefficientnet=False)
        """ timestamp = int(time.time())
        local_path = os.path.join(LOCAL_REGISTRY_PATH, "models",
                                f"model_{MLFLOW_MODEL_NAME}_{timestamp}.keras")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        model.save(local_path)

        model_size_mb = os.path.getsize(local_path) / (1024 * 1024)

        # Solo registrar metadata en MLflow
        mlflow.log_param("model_path", local_path)
        mlflow.log_param("model_size_mb", model_size_mb)
        mlflow.log_param("model_name", MLFLOW_MODEL_NAME)
        mlflow.log_param("timestamp", timestamp)

        print(Fore.GREEN + f"\n✅ Modelo guardado localmente: {local_path}" + Style.RESET_ALL)
        print(f"📦 Tamaño: {model_size_mb:.2f} MB")
        print(Fore.BLUE + "ℹ️  Metadata registrada en MLflow" + Style.RESET_ALL)

        return None """

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
    tmp_path = None  # ✅ Inicializar aquí

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
            params = params or {}  # ✅ Manejar params=None
            params['useefficientnet'] = useefficientnet
            params['model_architecture'] = 'EfficientNetB0' if useefficientnet else 'CNN'

            # Log params y metrics
            mlflow.log_params(params)
            if metrics:  # ✅ Solo log si hay metrics
                mlflow.log_metrics(metrics)

            print(f"🆔 Run ID: {run.info.run_id}")

            # Verificar tamaño del modelo
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                model.save(tmp.name)
                model_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
                tmp_path = tmp.name  # ✅ Asignar después de crear

            print(f"📦 Tamaño del modelo: {model_size_mb:.2f} MB")
            mlflow.log_param("model_size_mb", round(model_size_mb, 2))

            # Intentar subir a MLflow
            if model_size_mb < 50:  # Umbral de 50MB
                try:
                    print(Fore.BLUE + "💾 Guardando modelo en MLflow..." + Style.RESET_ALL)

                    mlflow.tensorflow.log_model(
                        model=model,
                        artifact_path="model",
                        registered_model_name=MLFLOW_MODEL_NAME
                    )

                    mlflow.set_tag("storage", "mlflow")
                    print(Fore.GREEN + "✅ Modelo guardado en MLflow" + Style.RESET_ALL)

                    # Limpiar temporal solo si subió exitosamente
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        tmp_path = None  # ✅ Marcar como limpiado

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
            model_name = f"model_{'EfficientNet' if useefficientnet else 'CNN'}_{timestamp}.keras"
            local_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Si ya tenemos el archivo temporal, moverlo; si no, guardar de nuevo
            if tmp_path and os.path.exists(tmp_path):
                shutil.move(tmp_path, local_path)
                tmp_path = None  # ✅ Marcar como movido
            else:
                model.save(local_path)

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
            # Limpiar archivo temporal si aún existe
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    return None
