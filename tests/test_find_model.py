#!/usr/bin/env python3
"""
Test específico para la función find_latest_model_by_architecture
"""

import sys
import os
sys.path.insert(0, '/Users/fernandorios/code/fermx3/coffee-disease-detection')

def test_find_latest_model():
    """Test la función find_latest_model_by_architecture"""

    from coffeedd.ml_logic.registry_ml import find_latest_model_by_architecture
    from coffeedd.params import LOCAL_REGISTRY_PATH

    models_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")

    print(f"🔍 Testing find_latest_model_by_architecture")
    print(f"📁 Base directory: {models_dir}")

    # Test 1: Buscar en todas las arquitecturas
    print("\n1. Buscando en todas las arquitecturas:")
    result = find_latest_model_by_architecture(models_dir)
    print(f"   Result: {result}")
    if result:
        print(f"   Type: {'File' if os.path.isfile(result) else 'Directory'}")
        print(f"   Exists: {os.path.exists(result)}")

    # Test 2: Buscar específicamente EfficientNet
    print("\n2. Buscando específicamente EfficientNet:")
    result_eff = find_latest_model_by_architecture(models_dir, 'efficientnet')
    print(f"   Result: {result_eff}")
    if result_eff:
        print(f"   Type: {'File' if os.path.isfile(result_eff) else 'Directory'}")
        print(f"   Exists: {os.path.exists(result_eff)}")

    # Test 3: Listar contenido de directorio efficientnet
    efficientnet_dir = os.path.join(models_dir, 'efficientnet')
    if os.path.exists(efficientnet_dir):
        print(f"\n3. Contenido de {efficientnet_dir}:")
        try:
            items = os.listdir(efficientnet_dir)
            for item in items:
                item_path = os.path.join(efficientnet_dir, item)
                item_type = "Dir" if os.path.isdir(item_path) else "File"
                size = ""
                if os.path.isfile(item_path):
                    size = f" ({os.path.getsize(item_path) / (1024*1024):.1f}MB)"
                print(f"   {item_type}: {item}{size}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"\n3. Directorio {efficientnet_dir} no existe")

    return result_eff

if __name__ == "__main__":
    print("🔧 Test de find_latest_model_by_architecture")
    print("=" * 60)

    result = test_find_latest_model()

    print("\n" + "=" * 60)
    if result and os.path.isfile(result):
        print("✅ Function working correctly - found file")
    elif result and os.path.isdir(result):
        print("⚠️  Function returning directory instead of file")
    else:
        print("❌ Function not finding models")
