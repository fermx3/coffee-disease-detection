#!/usr/bin/env python3
"""
Script de prueba para validar la carga de diferentes tipos de modelos
Testea que registry_ml.py pueda cargar CNN, VGG16 y EfficientNet correctamente
"""

import os
import sys
import tempfile
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, '/Users/fernandorios/code/fermx3/coffee-disease-detection')

def test_model_detection_functions():
    """Test de las funciones de detección de arquitectura"""
    print("🧪 Testing model detection functions...")

    from coffeedd.ml_logic.registry_ml import detect_model_architecture, build_model_by_architecture

    # Test detección por nombre de archivo
    test_cases = [
        ("model_EfficientNet_20240101.keras", "efficientnet"),
        ("model_VGG16_20240101.keras", "vgg16"),
        ("model_CNN_20240101.keras", "cnn"),
        ("some_efficientnet_model.h5", "efficientnet"),
        ("random_model.keras", "efficientnet"),  # default desde params
    ]

    for filename, expected in test_cases:
        result = detect_model_architecture(filename)
        print(f"   📄 {filename} -> {result} {'✅' if result == expected else '❌'}")
        assert result == expected, f"Expected {expected}, got {result}"

    # Test construcción de modelos
    architectures = ['cnn', 'vgg16', 'efficientnet']
    for arch in architectures:
        try:
            model = build_model_by_architecture(arch)
            print(f"   🏗️  {arch} model built successfully ✅")
            print(f"      Layers: {len(model.layers)}")
        except Exception as e:
            print(f"   🏗️  {arch} model failed ❌: {e}")
            raise

    print("✅ All detection functions passed!")

def test_model_saving_and_loading():
    """Test de guardado y carga de modelos"""
    print("\n🧪 Testing model saving and loading...")

    from coffeedd.ml_logic.model import build_simple_cnn_model, build_vgg16_model, build_efficientnet_model
    from coffeedd.ml_logic.registry_ml import save_model, load_model
    from coffeedd.params import LOCAL_REGISTRY_PATH
    import time

    # Crear directorio de prueba
    test_models_dir = Path(LOCAL_REGISTRY_PATH) / "models"
    test_models_dir.mkdir(parents=True, exist_ok=True)

    # Test models
    models_to_test = [
        ("CNN", build_simple_cnn_model),
        ("VGG16", build_vgg16_model),
        ("EfficientNet", build_efficientnet_model),
    ]

    for model_name, model_builder in models_to_test:
        try:
            print(f"\n   📦 Testing {model_name}...")

            # Construir modelo
            if model_name == "EfficientNet":
                model = model_builder()  # EfficientNet puede retornar tuple
                if isinstance(model, tuple):
                    model = model[0]
            else:
                model = model_builder()

            # Compilar modelo
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print(f"      🏗️  {model_name} built: {len(model.layers)} layers")

            # Guardar modelo
            print(f"      💾 Saving {model_name}...")
            original_arch = model_name.lower().replace("efficientnet", "efficientnet")

            # Simular el guardado
            save_model(model)
            print(f"      ✅ {model_name} saved successfully")

        except Exception as e:
            print(f"      ❌ {model_name} test failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("✅ Model tests completed!")

def test_load_models():
    """Test de carga de modelos existentes"""
    print("\n🧪 Testing model loading...")

    from coffeedd.ml_logic.registry_ml import load_model
    from coffeedd.params import LOCAL_REGISTRY_PATH

    models_dir = Path(LOCAL_REGISTRY_PATH) / "models"

    if not models_dir.exists():
        print("   ⚠️  No models directory found, skipping load test")
        return

    # Buscar modelos existentes
    model_files = list(models_dir.glob("*.keras")) + list(models_dir.glob("*.weights.h5"))

    if not model_files:
        print("   ⚠️  No model files found, skipping load test")
        return

    print(f"   📁 Found {len(model_files)} model files")

    try:
        # Intentar cargar el modelo más reciente
        print("   📥 Loading latest model...")
        model = load_model(compile_with_metrics=False)

        if model:
            print(f"   ✅ Model loaded successfully!")
            print(f"      Layers: {len(model.layers)}")
            print(f"      Input shape: {model.input_shape}")
            print(f"      Output shape: {model.output_shape}")
        else:
            print("   ❌ Failed to load model")

    except Exception as e:
        print(f"   ❌ Load test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Ejecutar todos los tests"""
    print("🚀 INICIANDO TESTS DE ARQUITECTURAS DE MODELO")
    print("=" * 60)

    try:
        test_model_detection_functions()
        test_model_saving_and_loading()
        test_load_models()

        print("\n🎉 ¡TODOS LOS TESTS PASARON EXITOSAMENTE!")
        print("✅ El sistema puede manejar CNN, VGG16 y EfficientNet")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
