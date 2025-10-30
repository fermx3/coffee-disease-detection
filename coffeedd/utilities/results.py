def combine_histories(history1, history2):
    """Combina dos historiales de Keras en uno solo."""
    combined = {}

    # Copiar history1
    for key, values in history1.history.items():
        combined[key] = values.copy()

    # Agregar history2
    for key, values in history2.history.items():
        if key in combined:
            combined[key].extend(values)
        else:
            combined[key] = values.copy()

    # Crear objeto compatible con Keras History
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
            self.epoch = list(range(len(history_dict[list(history_dict.keys())[0]])))

    return CombinedHistory(combined)
