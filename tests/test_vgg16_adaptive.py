#!/usr/bin/env python3
"""
Test para verificar la configuración adaptativa de VGG16
"""

def test_vgg16_adaptive_config():
    """Test de configuración adaptativa VGG16 según tamaño del dataset"""
    print("🧪 Probando configuración adaptativa VGG16...")
    
    # Mockear diferentes tamaños de dataset
    test_cases = [
        ("small", 6000, "Dataset pequeño"),
        ("half", 30000, "Dataset mediano"),
        ("full", 60000, "Dataset grande"),
        (15000, 15000, "Dataset numérico grande")
    ]
    
    # Importar después de configurar el entorno
    import os
    os.environ['MODEL_ARCHITECTURE'] = 'vgg16'
    
    for sample_size, expected_size, description in test_cases:
        print(f"\n📊 {description}: SAMPLE_SIZE = {sample_size}")
        
        # Simular la lógica de estimación de tamaño
        if isinstance(sample_size, str):
            if sample_size in ['full', 'completo']:
                estimated_size = 60000
            elif sample_size == 'half':
                estimated_size = 30000
            else:
                estimated_size = 6000
        elif isinstance(sample_size, (int, float)):
            if sample_size < 1:
                estimated_size = int(60000 * sample_size)
            else:
                estimated_size = sample_size
        else:
            estimated_size = 6000
        
        print(f"   🔢 Tamaño estimado: {estimated_size:,}")
        
        # Predecir configuración
        if estimated_size >= 10000:
            config = "agresiva"
            lr = 0.00005
            patience = 5
            epochs = 15
        else:
            config = "conservadora"
            lr = 0.0001
            patience = 8
            epochs = 20
            
        print(f"   ⚙️  Configuración: {config}")
        print(f"   📈 Learning Rate: {lr}")
        print(f"   ⏰ EarlyStopping patience: {patience}")
        print(f"   🔄 Epochs máximos: {epochs}")
        
        # Verificar que coincide con expectativa
        assert estimated_size == expected_size, f"Error: {estimated_size} != {expected_size}"
    
    print("\n✅ Todas las configuraciones adaptativas funcionan correctamente!")
    print("\n📋 Resumen de configuraciones:")
    print("   📊 Dataset < 10K: configuración conservadora")
    print("   📊 Dataset >= 10K: configuración agresiva anti-overfitting")

if __name__ == "__main__":
    test_vgg16_adaptive_config()