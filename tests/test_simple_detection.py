#!/usr/bin/env python3
"""
Test simple para verificar las funciones básicas
"""

import sys

sys.path.insert(0, "/Users/fernandorios/code/fermx3/coffee-disease-detection")


def test_basic_functions():
    print("Testing basic imports...")

    try:
        from coffeedd.ml_logic.registry_ml import (
            detect_model_architecture,
        )

        print("✅ Functions imported successfully")

        # Test detección básica
        result = detect_model_architecture("model_CNN_test.keras")
        print(f"CNN detection: {result}")

        result = detect_model_architecture("model_EfficientNet_test.keras")
        print(f"EfficientNet detection: {result}")

        result = detect_model_architecture("model_VGG16_test.keras")
        print(f"VGG16 detection: {result}")

        print("✅ All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functions()
