#!/usr/bin/env python3
"""
Script para probar el sistema de caché de modelos
"""

import os
import time

# Configurar entorno de prueba
os.environ["MODEL_ARCHITECTURE"] = "vgg16"
os.environ["MODEL_TARGET"] = "local"


def test_model_cache():
    """Prueba el sistema de caché de modelos"""

    print("🧪 Probando sistema de caché de modelos")
    print("=" * 50)

    from coffeedd.interface.main import get_cached_model, clear_model_cache

    # Test 1: Primera carga (debe cargar desde disco)
    print("\n1️⃣ Primera carga del modelo:")
    start_time = time.time()
    model1 = get_cached_model()
    load_time1 = time.time() - start_time

    if model1 is not None:
        print(f"   ✅ Modelo cargado en {load_time1:.2f}s")
        print(f"   📊 Capas: {len(model1.layers)}")
    else:
        print("   ❌ No se pudo cargar modelo")
        return

    # Test 2: Segunda carga (debe usar caché)
    print("\n2️⃣ Segunda carga del modelo (desde caché):")
    start_time = time.time()
    model2 = get_cached_model()
    load_time2 = time.time() - start_time

    if model2 is not None:
        print(f"   ✅ Modelo obtenido en {load_time2:.3f}s")
        print(f"   📊 Mismo objeto: {model1 is model2}")
        print(f"   ⚡ Mejora de velocidad: {(load_time1/load_time2):.1f}x más rápido")
    else:
        print("   ❌ Error en caché")

    # Test 3: Limpiar caché
    print("\n3️⃣ Limpiando caché:")
    clear_model_cache()

    # Test 4: Carga después de limpiar caché
    print("\n4️⃣ Carga después de limpiar caché:")
    start_time = time.time()
    model3 = get_cached_model()
    load_time3 = time.time() - start_time

    if model3 is not None:
        print(f"   ✅ Modelo recargado en {load_time3:.2f}s")
        print(f"   📊 Nuevo objeto: {model1 is not model3}")
    else:
        print("   ❌ Error al recargar")

    # Test 5: Múltiples cargas rápidas
    print("\n5️⃣ Test de múltiples cargas rápidas:")
    total_time = 0
    num_loads = 5

    for i in range(num_loads):
        start_time = time.time()
        model = get_cached_model()
        load_time = time.time() - start_time
        total_time += load_time
        print(f"   Carga {i+1}: {load_time:.3f}s")

    avg_time = total_time / num_loads
    print(f"   📊 Tiempo promedio: {avg_time:.3f}s")

    # Resumen
    print("\n📋 RESUMEN:")
    print(f"   • Primera carga: {load_time1:.2f}s (desde disco)")
    print(f"   • Carga desde caché: {load_time2:.3f}s")
    print(f"   • Recarga después de limpiar: {load_time3:.2f}s")
    print(f"   • Promedio cargas múltiples: {avg_time:.3f}s")
    print(f"   • Aceleración del caché: {(load_time1/load_time2):.1f}x")

    if load_time2 < load_time1 * 0.1:  # Al menos 10x más rápido
        print("   ✅ Caché funcionando correctamente")
    else:
        print("   ⚠️  Caché puede no estar funcionando óptimamente")


def test_cache_with_predictions():
    """Prueba el caché en el contexto de predicciones múltiples"""

    print("\n🔮 Probando caché en predicciones múltiples")
    print("=" * 50)

    from coffeedd.interface.main import pred
    import numpy as np
    from PIL import Image

    # Crear imagen dummy para prueba
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    # Múltiples predicciones para probar caché
    num_predictions = 3
    times = []

    for i in range(num_predictions):
        print(f"\n🔮 Predicción {i+1}:")
        start_time = time.time()

        try:
            result = pred(dummy_img)
            pred_time = time.time() - start_time
            times.append(pred_time)
            print(f"   ✅ Completada en {pred_time:.2f}s")
            print(f"   📊 Predicción: {result.get('predicted_class', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break

    if len(times) > 1:
        print("\n📊 Análisis de tiempos:")
        print(f"   • Primera predicción: {times[0]:.2f}s")
        print(f"   • Predicciones siguientes: {[f'{t:.2f}s' for t in times[1:]]}")

        if times[0] > max(times[1:]) * 1.5:  # Primera carga debe ser más lenta
            print("   ✅ Caché mejora tiempo en predicciones")
        else:
            print("   📝 Tiempos similares (modelo ya en caché)")


if __name__ == "__main__":
    try:
        test_model_cache()
        test_cache_with_predictions()
        print("\n🎉 Tests de caché completados!")
    except Exception as e:
        print(f"\n❌ Error en tests: {e}")
        import traceback

        traceback.print_exc()
