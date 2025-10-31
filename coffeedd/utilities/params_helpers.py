# Automatizar tipo de SAMPLE_SIZE
def auto_type(value):
    """Convierte automáticamente a int, float o string."""
    if value is None:
        return None
    value_str = str(value).strip()
    try:
        if '.' not in value_str:
            return int(value_str)
        return float(value_str)
    except ValueError:
        return value_str

# Determinar epochs basado en SAMPLE_SIZE
def get_epochs_for_sample_size(sample_size):
    """
    Retorna el número de epochs apropiado según el tamaño de muestra.

    Args:
        sample_size: Tamaño de muestra como int (número absoluto) o float (porcentaje 0-1)

    Returns:
        int: Número de epochs recomendado

    Examples:
        >>> get_epochs_for_sample_size(0.1)
        30
        >>> get_epochs_for_sample_size(100)
        30
        >>> get_epochs_for_sample_size(1000)
        40
        >>> get_epochs_for_sample_size(1.0)
        60
    """

    # Validar tipo
    if not isinstance(sample_size, (int, float)):
        print(f"⚠️  SAMPLE_SIZE debe ser int o float, recibido: {type(sample_size).__name__}")
        return 60

    # Si es float (porcentaje)
    if isinstance(sample_size, float):
        # Validar rango para floats (debe estar entre 0 y 1 para porcentajes)
        if 0 < sample_size <= 1:
            if sample_size <= 0.1:
                return 30
            elif sample_size <= 0.5:
                return 50
            else:
                return 60
        else:
            # Float fuera de rango válido
            print(f"⚠️  SAMPLE_SIZE float debe estar entre 0 y 1, recibido: {sample_size}")
            return 60

    # Si es int (número absoluto)
    elif isinstance(sample_size, int):
        # Validar que sea positivo
        if sample_size <= 0:
            print(f"⚠️  SAMPLE_SIZE debe ser positivo, recibido: {sample_size}")
            return 60

        if sample_size <= 100:
            return 30
        elif sample_size <= 1000:
            return 40
        elif sample_size <= 5000:
            return 50
        else:
            return 60

    # Fallback (no debería llegar aquí)
    return 60

def get_sample_name(sample_size):
    """
    Genera un nombre descriptivo basado en el tamaño de muestra.

    Args:
        sample_size: Tamaño de muestra (int, float, str, o None)

    Returns:
        str: Nombre descriptivo para usar en archivos y logs

    Examples:
        >>> get_sample_name(None)
        'full'
        >>> get_sample_name('full')
        'full'
        >>> get_sample_name('half')
        'half'
        >>> get_sample_name(1.0)
        '100pct'
        >>> get_sample_name(0.5)
        '50pct'
        >>> get_sample_name(0.1)
        '10pct'
        >>> get_sample_name(100)
        '100'
        >>> get_sample_name(1000)
        '1000'
    """
    # Casos None o 'full'
    if sample_size is None or sample_size == 'full':
        return 'full'

    # Caso especial 'half'
    if sample_size == 'half':
        return 'half'

    # Si es float (porcentaje), convertir a formato "XXpct"
    if isinstance(sample_size, float):
        # Asegurar que el porcentaje esté entre 0 y 1
        if 0 < sample_size <= 1:
            percentage = int(sample_size * 100)
            return f'{percentage}pct'
        else:
            # Si el float es mayor a 1, tratarlo como número absoluto
            return str(int(sample_size))

    # Si es int o cualquier otro tipo, convertir a string
    return str(sample_size)
