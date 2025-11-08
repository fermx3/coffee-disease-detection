"""
Funciones para subir modelos a Google Cloud Storage
"""

import os
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from colorama import Fore, Style
import json

from coffeedd.params import (
    GCP_PROJECT,
    BUCKET_NAME,
    MODEL_ARCHITECTURE,
    LOCAL_REGISTRY_PATH,
)


def upload_latest_model_to_gcs(
    model_version: str = None, include_metadata: bool = True, dry_run: bool = False
) -> dict:
    """
    Sube el último modelo entrenado a Google Cloud Storage

    Args:
        model_version: Versión específica del modelo (si None, usa timestamp)
        include_metadata: Si incluir metadatos del modelo
        dry_run: Si es True, solo simula la subida sin ejecutarla

    Returns:
        dict: Información sobre la subida (rutas, tamaño, etc.)
    """

    print(Fore.CYAN + "\n☁️  SUBIENDO MODELO A GOOGLE CLOUD STORAGE" + Style.RESET_ALL)
    print("=" * 60)

    # 1. Verificar configuración GCS
    if not _verify_gcs_config():
        raise ValueError("Configuración de GCS incompleta. Revisa params.py")

    # 2. Encontrar el último modelo
    latest_model_path = _find_latest_model()
    if not latest_model_path:
        raise FileNotFoundError("No se encontró ningún modelo entrenado")

    print(f"📁 Modelo encontrado: {latest_model_path}")

    # 3. Preparar metadatos
    metadata = {}
    if include_metadata:
        metadata = _collect_model_metadata(latest_model_path)
        print(f"📊 Metadatos recolectados: {len(metadata)} campos")

    # 4. Generar versión del modelo
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 5. Generar nombre consistente del modelo
    model_type = metadata.get("model_type", "CNN") if metadata else "CNN"
    model_filename = f"model_{model_type}_{model_version}.keras"

    # Detectar arquitectura para la estructura de carpetas
    architecture = None
    if metadata and "architecture" in metadata:
        architecture = metadata["architecture"]
    elif metadata and "model_type" in metadata:
        # Compatibilidad con metadatos antiguos
        if metadata["model_type"] == "EfficientNet":
            architecture = "efficientnet"
        elif metadata["model_type"] == "VGG16":
            architecture = "vgg16"
        else:
            architecture = "cnn"

    # 6. Definir rutas en GCS con arquitectura específica
    gcs_paths = _generate_gcs_paths(model_version, model_filename, architecture)

    # 7. Mostrar resumen antes de subir
    model_size = _get_file_size(latest_model_path)
    print("\n📋 Resumen de subida:")
    print(f"   • Archivo local: {latest_model_path}")
    print(f"   • Tamaño: {model_size:.2f} MB")
    print(f"   • Bucket: {BUCKET_NAME}")
    print(f"   • Versión: {model_version}")
    print(f"   • Ruta GCS modelo: {gcs_paths['model']}")
    if metadata:
        print(f"   • Ruta GCS metadata: {gcs_paths['metadata']}")

    if dry_run:
        print(f"\n{Fore.YELLOW}🔍 DRY RUN - No se subirá realmente{Style.RESET_ALL}")
        return {
            "success": True,
            "dry_run": True,
            "model_version": model_version,
            "local_path": str(latest_model_path),
            "gcs_paths": gcs_paths,
            "model_size_mb": model_size,
            "metadata_fields": len(metadata),
        }

    # 7. Subir a GCS
    upload_results = _upload_to_gcs(latest_model_path, metadata, gcs_paths)

    # 8. Verificar subida
    verification_results = _verify_upload(gcs_paths)

    print(f"\n{Fore.GREEN}✅ MODELO SUBIDO EXITOSAMENTE{Style.RESET_ALL}")
    print(f"   • Modelo: gs://{BUCKET_NAME}/{gcs_paths['model']}")
    if metadata:
        print(f"   • Metadata: gs://{BUCKET_NAME}/{gcs_paths['metadata']}")
    print(f"   • Versión: {model_version}")

    return {
        "success": True,
        "dry_run": False,
        "model_version": model_version,
        "local_path": str(latest_model_path),
        "gcs_paths": gcs_paths,
        "model_size_mb": model_size,
        "upload_results": upload_results,
        "verification_results": verification_results,
        "metadata_fields": len(metadata),
    }


def _verify_gcs_config() -> bool:
    """
    Verifica que la configuración de GCS esté completa

    Returns:
        bool: True si toda la configuración está presente, False si no
    """

    # Variables de entorno requeridas
    required_vars = ["GCP_PROJECT", "BUCKET_NAME"]

    print("🔍 DEBUG: Verificando variables de entorno...")

    # Verificar que todas las variables existan y no estén vacías
    for var in required_vars:
        value = os.environ.get(var)
        print(f"   {var}: {value}")

        if not value or value.strip() == "":
            print(f"⚠️ Variable de entorno faltante o vacía: {var}")
            return False

    print("✅ Variables de entorno configuradas correctamente")

    # Verificar que podemos conectar a GCS (solo si las variables están)
    try:
        from google.cloud import storage

        project_id = os.environ.get("GCP_PROJECT")
        bucket_name = os.environ.get("BUCKET_NAME")

        print("🔍 DEBUG: Intentando conectar a GCS...")
        print(f"   Project: {project_id}")
        print(f"   Bucket: {bucket_name}")

        # Intentar crear cliente
        client = storage.Client(project=project_id)

        # Verificar que el bucket existe
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            print(f"⚠️ Bucket {bucket_name} no existe o no es accesible")
            return False

        print(f"✅ Conexión a GCS verificada: {bucket_name}")
        return True

    except Exception as e:
        print(f"⚠️ Error verificando configuración GCS: {e}")
        return False


def _find_latest_model() -> Path:
    """Encuentra el archivo de modelo más reciente, buscando en carpetas de arquitectura"""
    models_base_dir = Path(LOCAL_REGISTRY_PATH, "models")

    if not models_base_dir.exists():
        return None

    model_files = []

    # Buscar en directorio base (compatibilidad con modelos antiguos)
    for pattern in ["*.h5", "*.keras", "*.weights.h5"]:
        model_files.extend(list(models_base_dir.glob(pattern)))

    # Buscar en subcarpetas de arquitectura
    for arch_folder in ["cnn", "vgg16", "efficientnet"]:
        arch_dir = models_base_dir / arch_folder
        if arch_dir.exists():
            for pattern in ["*.h5", "*.keras", "*.weights.h5"]:
                model_files.extend(list(arch_dir.glob(pattern)))

    if not model_files:
        return None

    # Retornar el más reciente por fecha de modificación
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_model


def _collect_model_metadata(model_path: Path) -> dict:
    """Recolecta metadatos del modelo y entrenamiento"""

    # Detectar tipo de modelo desde el archivo
    model_type = "CNN"  # por defecto

    # Intentar cargar el modelo para detectar su tipo
    try:
        import keras

        if model_path.suffix == ".keras":
            # Cargar solo la configuración para detectar tipo
            temp_model = keras.models.load_model(model_path, compile=False)
            if any("efficientnet" in layer.name.lower() for layer in temp_model.layers):
                model_type = "EfficientNet"
        elif model_path.suffix == ".h5":
            # Para archivos .h5, inferir del nombre si es posible
            if "efficientnet" in model_path.name.lower():
                model_type = "EfficientNet"

        # Buscar archivos de configuración que puedan tener la info
        config_files = list(model_path.parent.glob("*config.json"))
        for config_file in config_files:
            try:
                import json

                with open(config_file, "r") as f:
                    config_data = json.load(f)
                    if config_data.get("model_type") == "EfficientNet":
                        model_type = "EfficientNet"
                        break
            except:
                continue

    except Exception as e:
        print(f"⚠️  No se pudo detectar tipo de modelo, usando CNN por defecto: {e}")

    metadata = {
        "model_filename": model_path.name,
        "model_size_bytes": model_path.stat().st_size,
        "created_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        "upload_timestamp": datetime.now().isoformat(),
        "model_name": MODEL_ARCHITECTURE,
        "model_type": model_type,
    }

    # Buscar archivos de registro relacionados
    models_dir = model_path.parent

    # Buscar archivos de métricas
    metrics_files = list(models_dir.glob("*metrics*.json"))
    if metrics_files:
        latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_metrics, "r") as f:
                metrics_data = json.load(f)
                metadata["training_metrics"] = metrics_data
                metadata["metrics_file"] = latest_metrics.name
        except Exception as e:
            print(f"⚠️  Error leyendo métricas: {e}")

    # Información del registry local si existe
    registry_path = Path(LOCAL_REGISTRY_PATH)
    if registry_path.exists():
        registry_files = list(registry_path.glob("*.csv"))
        if registry_files:
            latest_registry = max(registry_files, key=lambda p: p.stat().st_mtime)
            metadata["registry_file"] = latest_registry.name

    return metadata


def _generate_gcs_paths(
    model_version: str, model_filename: str, architecture: str = None
) -> dict:
    """Genera las rutas en GCS para modelo y metadatos"""
    # Usar arquitectura específica del modelo, o la configurada por defecto
    arch_for_path = architecture or MODEL_ARCHITECTURE.lower()

    base_path = f"models/{arch_for_path}"

    return {
        "model": f"{base_path}/v{model_version}/{model_filename}",
        "metadata": f"{base_path}/v{model_version}/metadata.json",
        "version_folder": f"{base_path}/v{model_version}/",
    }


def _get_file_size(file_path: Path) -> float:
    """Retorna el tamaño del archivo en MB"""
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def _upload_to_gcs(model_path: Path, metadata: dict, gcs_paths: dict) -> dict:
    """Ejecuta la subida real a GCS"""
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)

    results = {}

    try:
        # Subir modelo
        print("📤 Subiendo modelo...")
        model_blob = bucket.blob(gcs_paths["model"])
        model_blob.upload_from_filename(str(model_path))
        results["model_uploaded"] = True
        results["model_gcs_path"] = f"gs://{BUCKET_NAME}/{gcs_paths['model']}"

        # Subir metadatos si existen
        if metadata:
            print("📤 Subiendo metadatos...")
            metadata_blob = bucket.blob(gcs_paths["metadata"])
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2), content_type="application/json"
            )
            results["metadata_uploaded"] = True
            results["metadata_gcs_path"] = f"gs://{BUCKET_NAME}/{gcs_paths['metadata']}"

        return results

    except Exception as e:
        print(f"{Fore.RED}❌ Error en subida: {e}{Style.RESET_ALL}")
        raise


def _verify_upload(gcs_paths: dict) -> dict:
    """Verifica que los archivos se subieron correctamente"""
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)

    verification = {}

    # Verificar modelo
    model_blob = bucket.blob(gcs_paths["model"])
    verification["model_exists"] = model_blob.exists()
    if verification["model_exists"]:
        verification["model_size"] = model_blob.size

    # Verificar metadatos
    metadata_blob = bucket.blob(gcs_paths["metadata"])
    verification["metadata_exists"] = metadata_blob.exists()
    if verification["metadata_exists"]:
        verification["metadata_size"] = metadata_blob.size

    return verification


def list_models_in_gcs(limit: int = 10) -> list:
    """Lista los modelos subidos a GCS"""
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)

    prefix = f"models/{MODEL_ARCHITECTURE}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    models = []
    for blob in blobs:
        if blob.name.endswith(".h5") or blob.name.endswith(".keras"):
            models.append(
                {
                    "name": blob.name,
                    "size_mb": blob.size / (1024 * 1024),
                    "created": blob.time_created,
                    "updated": blob.updated,
                    "gcs_path": f"gs://{BUCKET_NAME}/{blob.name}",
                }
            )

    # Ordenar por fecha de creación (más reciente primero)
    models.sort(key=lambda x: x["created"], reverse=True)

    return models[:limit]


if __name__ == "__main__":
    # Test básico
    try:
        result = upload_latest_model_to_gcs(dry_run=True)
        print(f"✅ Test completado: {result}")
    except Exception as e:
        print(f"❌ Error en test: {e}")
