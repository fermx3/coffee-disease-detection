from keras.models import Model


def is_efficientnet_model(model: Model) -> bool:
    """Detecta si el modelo usa EfficientNet analizando su arquitectura."""
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            return True
    return False
