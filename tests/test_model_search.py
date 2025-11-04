#!/usr/bin/env python3
"""
Test para verificar búsqueda de modelos local y GCS
"""

import sys
import os
sys.path.insert(0, '/Users/fernandorios/code/fermx3/coffee-disease-detection')

def test_local_model_search():
    """Test de búsqueda local de modelos"""
    print("🔍 Testing local model search...")

    try:
        from coffeedd.ml_logic.registry_ml import find_latest_model_by_architecture
        from coffeedd.params import LOCAL_REGISTRY_PATH

        models_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")

        print(f"📁 Searching in: {models_dir}")

        # Test 1: Búsqueda general (todas las arquitecturas)
        print("\n1. Búsqueda general:")
        latest_general = find_latest_model_by_architecture(models_dir)
        if latest_general:
            print(f"   ✅ Encontrado: {os.path.basename(latest_general)}")
            print(f"   📁 Ruta completa: {latest_general}")
            print(f"   📊 Tamaño: {os.path.getsize(latest_general) / (1024*1024):.1f}MB")
        else:
            print("   ❌ No se encontraron modelos")

        # Test 2: Búsqueda por arquitectura específica
        architectures = ['cnn', 'vgg16', 'efficientnet']
        for arch in architectures:
            print(f"\n2. Búsqueda específica - {arch}:")
            latest_arch = find_latest_model_by_architecture(models_dir, arch)
            if latest_arch:
                print(f"   ✅ Encontrado: {os.path.basename(latest_arch)}")
                print(f"   📁 Ruta completa: {latest_arch}")
                print(f"   📊 Tamaño: {os.path.getsize(latest_arch) / (1024*1024):.1f}MB")
            else:
                print(f"   ⚠️  No se encontraron modelos para {arch}")

        return True

    except Exception as e:
        print(f"❌ Error en búsqueda local: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gcs_model_search():
    """Test de búsqueda GCS de modelos (solo si está configurado)"""
    print("\n☁️  Testing GCS model search...")

    try:
        from coffeedd.params import MODEL_TARGET, BUCKET_NAME, GCP_PROJECT

        if MODEL_TARGET != "gcs":
            print("   ⚠️  MODEL_TARGET no es GCS, saltando test")
            return True

        if not BUCKET_NAME or not GCP_PROJECT:
            print("   ⚠️  Configuración GCS incompleta, saltando test")
            return True

        from google.cloud import storage

        print(f"🪣 Bucket: {BUCKET_NAME}")
        print(f"🏗️  Project: {GCP_PROJECT}")

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)

        # Listar modelos en las diferentes estructuras
        prefixes_to_test = [
            "models/",
            "models/cnn/",
            "models/vgg16/",
            "models/efficientnet/",
            "models/cnn/v",
            "models/vgg16/v",
            "models/efficientnet/v"
        ]

        all_models = []

        for prefix in prefixes_to_test:
            print(f"\n   🔍 Buscando en: {prefix}")
            blobs = list(bucket.list_blobs(prefix=prefix))

            model_blobs = []
            for blob in blobs:
                if not blob.name.endswith('/'):
                    if blob.name.endswith(('.keras', '.h5', '.weights.h5')):
                        if not blob.name.endswith('_config.json'):
                            if blob.size and blob.size > 1024 * 1024:  # > 1MB
                                model_blobs.append(blob)

            if model_blobs:
                print(f"      📦 Encontrados {len(model_blobs)} modelos:")
                for blob in model_blobs[:3]:  # Mostrar solo los primeros 3
                    print(f"         - {blob.name} ({blob.size / (1024*1024):.1f}MB)")
                if len(model_blobs) > 3:
                    print(f"         ... y {len(model_blobs) - 3} más")
                all_models.extend(model_blobs)
            else:
                print("      📭 No se encontraron modelos")

        print(f"\n📊 Total de modelos encontrados en GCS: {len(all_models)}")

        if all_models:
            # Encontrar el más reciente
            latest = max(all_models, key=lambda x: x.updated)
            print(f"🎯 Modelo más reciente: {latest.name}")
            print(f"   📅 Actualizado: {latest.updated}")
            print(f"   📊 Tamaño: {latest.size / (1024*1024):.1f}MB")

        return True

    except Exception as e:
        print(f"❌ Error en búsqueda GCS: {e}")
        print("   💡 Verifica configuración de GCP y credenciales")
        return False

def test_model_loading():
    """Test de carga completa de modelo"""
    print("\n🔄 Testing complete model loading...")

    try:
        from coffeedd.ml_logic.registry_ml import load_model

        model = load_model()

        if model is None:
            print("   ⚠️  No se pudo cargar modelo (normal si no hay modelos)")
            return True
        else:
            print(f"   ✅ Modelo cargado exitosamente")
            print(f"   📊 Capas: {len(model.layers)}")
            print(f"   🏗️  Tipo: {type(model).__name__}")
            return True

    except Exception as e:
        print(f"   ❌ Error en carga de modelo: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Test de búsqueda y carga de modelos")
    print("=" * 60)

    success_local = test_local_model_search()
    success_gcs = test_gcs_model_search()
    success_loading = test_model_loading()

    print("\n" + "=" * 60)
    print("📋 Resumen de tests:")
    print(f"   🔍 Búsqueda local: {'✅' if success_local else '❌'}")
    print(f"   ☁️  Búsqueda GCS: {'✅' if success_gcs else '❌'}")
    print(f"   🔄 Carga de modelo: {'✅' if success_loading else '❌'}")

    overall_success = success_local and success_gcs and success_loading
    print(f"\n🎯 Estado general: {'✅ TODOS LOS TESTS PASARON' if overall_success else '❌ ALGUNOS TESTS FALLARON'}")

    if overall_success:
        print("💡 El sistema de modelos está funcionando correctamente")
    else:
        print("💡 Revisa los errores arriba para solucionar problemas")

    sys.exit(0 if overall_success else 1)
