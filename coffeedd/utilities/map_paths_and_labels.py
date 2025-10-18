from pathlib import Path
from typing import List, Tuple, Sequence, Optional


from coffeedd.params import *


def map_paths_and_labels(
    data_dir: Path | str,
    recursive: bool = False,
    patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Genera listas de rutas y etiquetas desde un directorio con subcarpetas por clase.

    Parámetros:
    - data_dir: ruta raíz que contiene una carpeta por clase.
    - recursive: si True busca recursivamente dentro de cada clase.
    - patterns: secuencia de glob patterns a incluir (por defecto extensiones comunes de imagen).

    Retorna:
    - image_paths: lista de rutas de imágenes (str).
    - labels: lista de enteros con el índice de la clase.
    - class_names: lista de nombres de clase (orden alfabético) usada para mapear labels.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directorio inválido: {root}")

    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not class_names:
        return [], [], []

    pats = tuple(patterns) if patterns else IMG_GLOBS
    image_paths: List[str] = []
    labels: List[int] = []

    for idx, cls in enumerate(class_names):
        cdir = root / cls
        files = []
        for pat in pats:
            it = cdir.rglob(pat) if recursive else cdir.glob(pat)
            files.extend([p for p in it if p.is_file()])
        # Orden determinista y sin duplicados
        for p in sorted(set(files)):
            image_paths.append(str(p))
            labels.append(idx)

    return image_paths, labels, class_names


if __name__ == "__main__":
    # Ruta a la carpeta principal
    DATA_DIR = Path(LOCAL_DATA_PATH) / "train"

    # Lista de clases (ordenadas alfabéticamente por defecto)
    class_names = sorted([item.name for item in DATA_DIR.glob("*") if item.is_dir()])
    print("Clases:", class_names)

    # Mapear paths y labels
    all_image_paths = []
    all_labels = []

    for idx, class_name in enumerate(class_names):
        folder = DATA_DIR / class_name
        for path in folder.glob("*.*"):  # acepta jpg, png, etc.
            all_image_paths.append(str(path))
            all_labels.append(idx)

    print(f"Total imágenes: {len(all_image_paths)}")
