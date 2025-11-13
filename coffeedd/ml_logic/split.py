"""Funciones para dividir datasets de imágenes en conjuntos de entrenamiento, validación y prueba."""

from pathlib import Path
import shutil
import random
from typing import Dict, List, Tuple, Optional

from coffeedd.params import *

# Intenta obtener rutas por defecto desde params.py
if "LOCAL_RAW_DATA_PATH" in globals() and "LOCAL_DATA_PATH" in globals():
    LOCAL_RAW_DATA_PATH = Path(LOCAL_RAW_DATA_PATH)
    LOCAL_PROCESSED_DATA_PATH = Path(LOCAL_DATA_PATH)
else:
    # Fallback si no existe params o variables
    LOCAL_RAW_DATA_PATH = Path("data/raw_data")
    LOCAL_PROCESSED_DATA_PATH = Path("data/process_data")


def _gather_images(
    folder: Path, patterns: Tuple[str, ...], recursive: bool = True
) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        it = folder.rglob(pat) if recursive else folder.glob(pat)
        files.extend([p for p in it if p.is_file()])
    return files


def split_dataset(
    src_root: Optional[Path | str] = None,
    output_root: Optional[Path | str] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    move: bool = False,
    seed: Optional[int] = 42,
    patterns: Tuple[str, ...] = IMG_PATTERNS,
    clean_output: bool = False,
    recursive: bool = True,
) -> Dict:
    """
    Divide un dataset de imágenes organizado por carpetas de clase en train/val/test.

    - src_root: raíz que contiene subcarpetas por clase (por defecto LOCAL_RAW_DATA_PATH).
    - output_root: raíz donde se crearán train/, val/, test/ (por defecto LOCAL_PROCESSED_DATA_PATH/'splits').
    - test_size y val_size: fracciones del total (0-1).
    - move: si True mueve archivos; si False copia.
    - seed: semilla para reproducibilidad.
    - patterns: extensiones a considerar como imágenes.
    - clean_output: si True borra la carpeta de salida antes de crearla.
    - recursive: si True busca imágenes de forma recursiva dentro de cada clase.
    """
    if src_root is None:
        src_root = LOCAL_RAW_DATA_PATH
    if output_root is None:
        output_root = LOCAL_PROCESSED_DATA_PATH

    src_root = Path(src_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Directorio de origen no válido: {src_root}")

    if src_root == output_root or str(output_root).startswith(str(src_root)):
        raise ValueError(
            "output_root no debe estar dentro de src_root (evita contaminar el origen)."
        )

    # Limpieza opcional
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir(parents=True, exist_ok=True)

    # Detectar clases: subcarpetas directas (excluye nombres de splits)
    excluded = {"train", "val", "test", ".DS_Store"}
    class_dirs = [
        d for d in src_root.iterdir() if d.is_dir() and d.name not in excluded
    ]
    if not class_dirs:
        raise ValueError(f"No se encontraron subcarpetas de clases en {src_root}")

    rng = random.Random(seed)

    summary = {
        "root": str(src_root),
        "output": str(output_root),
        "params": {
            "test_size": test_size,
            "val_size": val_size,
            "move": move,
            "seed": seed,
        },
        "splits": {"train": 0, "val": 0, "test": 0},
        "classes": {},
    }

    for cdir in class_dirs:
        cls = cdir.name
        files = _gather_images(cdir, patterns, recursive=recursive)
        if not files:
            continue

        idx = list(range(len(files)))
        rng.shuffle(idx)

        n = len(files)
        n_test = max(0, int(n * test_size))
        n_val = max(0, int(n * val_size))

        # Garantiza que haya al menos 1 en train si hay imágenes
        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        test_idx = set(idx[:n_test])
        val_idx = set(idx[n_test : n_test + n_val])
        train_idx = set(idx[n_test + n_val :])

        # Crear carpetas destino por clase
        for split in ("train", "val", "test"):
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)

        def transfer(p: Path, dst: Path):
            dst_parent = dst.parent
            dst_parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))

        # Transferir
        moved_class = {"train": 0, "val": 0, "test": 0}
        for i, p in enumerate(files):
            if i in test_idx:
                dst = output_root / "test" / cls / p.name
                transfer(p, dst)
                moved_class["test"] += 1
                summary["splits"]["test"] += 1
            elif i in val_idx:
                dst = output_root / "val" / cls / p.name
                transfer(p, dst)
                moved_class["val"] += 1
                summary["splits"]["val"] += 1
            else:
                dst = output_root / "train" / cls / p.name
                transfer(p, dst)
                moved_class["train"] += 1
                summary["splits"]["train"] += 1

        summary["classes"][cls] = moved_class

    return summary
