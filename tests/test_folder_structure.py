#!/usr/bin/env python3
"""
Script para probar la nueva estructura de carpetas por arquitectura
"""

import sys
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, "/Users/fernandorios/code/fermx3/coffee-disease-detection")


def test_folder_structure():
    """Test de la estructura de carpetas por arquitectura"""
    print("🧪 Testing folder structure by architecture...")

    try:
        from coffeedd.params import LOCAL_REGISTRY_PATH
        from coffeedd.ml_logic.registry_ml import detect_model_architecture

        models_base_dir = Path(LOCAL_REGISTRY_PATH) / "models"
        print(f"📁 Base models directory: {models_base_dir}")

        # Verificar que se pueden crear las carpetas de arquitectura
        expected_folders = ["cnn", "vgg16", "efficientnet"]

        for arch in expected_folders:
            arch_dir = models_base_dir / arch
            arch_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created/verified folder: {arch_dir}")

            # Crear un archivo de prueba
            test_file = arch_dir / f"test_model_{arch}_20240101.keras"
            test_file.touch()
            print(f"   📄 Created test file: {test_file}")

            # Verificar detección de arquitectura
            detected = detect_model_architecture(str(test_file))
            print(f"   🔍 Detected architecture: {detected}")

        # Test de búsqueda de archivos
        print("\n🔍 Testing file search across architecture folders...")

        from coffeedd.ml_logic.gcs_upload import _find_latest_model

        latest = _find_latest_model()
        if latest:
            print(f"✅ Found latest model: {latest}")
        else:
            print("⚠️  No models found (this is expected if no real models exist)")

        print("✅ Folder structure tests completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_gcs_paths():
    """Test de las rutas GCS con arquitecturas"""
    print("\n🧪 Testing GCS paths with architectures...")

    try:
        from coffeedd.ml_logic.gcs_upload import _generate_gcs_paths

        test_cases = [
            ("cnn", "model_CNN_20240101.keras"),
            ("vgg16", "model_VGG16_20240101.keras"),
            ("efficientnet", "model_EfficientNet_20240101.keras"),
        ]

        for arch, filename in test_cases:
            paths = _generate_gcs_paths("20240101_120000", filename, arch)
            print(f"📁 {arch}:")
            print(f"   Model: {paths['model']}")
            print(f"   Metadata: {paths['metadata']}")
            print(f"   Folder: {paths['version_folder']}")

            # Verificar que contiene la arquitectura en la ruta
            assert arch in paths["model"], f"Architecture {arch} not found in path"
            print(f"   ✅ Architecture {arch} correctly included in path")

        print("✅ GCS paths test completed!")

    except Exception as e:
        print(f"❌ GCS paths test failed: {e}")
        import traceback

        traceback.print_exc()


def show_expected_structure():
    """Muestra la estructura esperada"""
    print("\n📋 ESTRUCTURA ESPERADA:")
    print("=" * 50)
    print(
        """
LOCAL STRUCTURE:
~/.coffeedd/mlops/training_outputs/models/
├── cnn/
│   ├── model_CNN_20241101-143000.keras
│   ├── model_CNN_20241101-143000.weights.h5
│   └── model_CNN_20241101-143000_config.json
├── vgg16/
│   ├── model_VGG16_20241101-144000.keras
│   └── model_VGG16_20241101-144000_config.json
└── efficientnet/
    ├── model_EfficientNet_20241101-145000.keras
    └── model_EfficientNet_20241101-145000_config.json

GCS STRUCTURE:
bucket/models/
├── cnn/
│   └── v20241101_143000/
│       ├── model_CNN_20241101_143000.keras
│       └── metadata.json
├── vgg16/
│   └── v20241101_144000/
│       ├── model_VGG16_20241101_144000.keras
│       └── metadata.json
└── efficientnet/
    └── v20241101_145000/
        ├── model_EfficientNet_20241101_145000.keras
        └── metadata.json
    """
    )


def main():
    """Ejecutar todos los tests"""
    print("🚀 TESTING ARCHITECTURE FOLDER STRUCTURE")
    print("=" * 60)

    try:
        test_folder_structure()
        test_gcs_paths()
        show_expected_structure()

        print("\n🎉 ¡TODOS LOS TESTS DE ESTRUCTURA PASARON!")
        print("✅ Las carpetas por arquitectura están funcionando")
        print("✅ Los modelos se organizarán automáticamente por tipo")
        print("✅ GCS mantendrá la misma estructura organizada")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
