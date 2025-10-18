"""
Divide imágenes en subcarpetas según una columna de un archivo Excel.
Lee un archivo Excel (.xlsx) que contiene nombres de archivos y sus
subclases, busca las imágenes en una carpeta raíz y las copia o mueve
a subcarpetas nombradas según la subclase.
Desarrollado para RoCoLe Dataset.
Uso:
    python image-organizer.py --excel datos.xlsx --src /ruta/a/imagenes --dest /ruta/a/destino
Opciones:
    --excel/-x: Ruta al archivo Excel (.xlsx)
    --src/-s: Carpeta raíz donde buscar las imágenes
    --dest/-d: Carpeta destino donde crear subcarpetas por subclass
    --file-col: Nombre de la columna que contiene el nombre de archivo (por defecto: File)
    --subclass-col: Nombre de la columna con la subclase (por defecto: Multiclass.Label)
    --sheet: Nombre o índice de la hoja en el Excel (opcional)
    --move: Mover archivos en vez de copiarlos
"""

import argparse
import shutil
from pathlib import Path
import sys

try:
    import pandas as pd
except Exception as e:
    print("Error: pandas is required. Install with: pip install pandas openpyxl")
    raise e


def find_file(root: Path, filename: str) -> Path | None:
    """
    Busca un archivo dentro de root (recursivo). Intenta coincidencia exacta,
    luego por nombre sin extensión (stem) y case-insensitive.
    """
    target = Path(filename)
    # Si ya es path absoluto/relativo válido
    if target.is_file():
        return target

    name = target.name
    # búsqueda exacta por nombre
    for p in root.rglob(name):
        if p.is_file():
            return p

    # búsqueda por stem (sin extensión), case-insensitive
    stem = Path(name).stem.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.stem.lower() == stem:
            return p

    return None


def divide_images(
    excel_path: Path,
    src_dir: Path,
    dest_dir: Path,
    file_col: str = "File",
    subclass_col: str = "Multiclass.Label",
    sheet_name: str | None = None,
    move: bool = False,
) -> None:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)
    if file_col not in df.columns or subclass_col not in df.columns:
        print(
            f"Columnas esperadas no encontradas en el Excel. Columnas: {list(df.columns)}"
        )
        sys.exit(1)

    src_dir = src_dir.expanduser().resolve()
    dest_dir = dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    moved = 0
    missing = 0

    for _, row in df.iterrows():
        fname = row.get(file_col)
        subclass = row.get(subclass_col)

        if pd.isna(fname) or pd.isna(subclass):
            continue

        fname = str(fname).strip()
        subclass = str(subclass).strip()

        total += 1
        found = find_file(src_dir, fname)
        if not found:
            print(f"[MISSING] {fname}")
            missing += 1
            continue

        target_dir = dest_dir / subclass
        target_dir.mkdir(parents=True, exist_ok=True)

        dest_path = target_dir / found.name
        try:
            if move:
                shutil.move(str(found), str(dest_path))
            else:
                shutil.copy2(str(found), str(dest_path))
            moved += 1
            print(
                f"[OK] {found.relative_to(src_dir)} -> {target_dir.name}/{found.name}"
            )
        except Exception as e:
            print(f"[ERROR] al copiar/mover {found}: {e}")

    print(
        f"\nResumen: total filas procesadas={total}, copiados/movidos={moved}, faltantes={missing}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Dividir imágenes en subcarpetas según columna 'Multiclass.Label' en un Excel."
    )
    p.add_argument("--excel", "-x", required=True, help="Ruta al archivo Excel (.xlsx)")
    p.add_argument(
        "--src", "-s", required=True, help="Carpeta raíz donde buscar las imágenes"
    )
    p.add_argument(
        "--dest",
        "-d",
        required=True,
        help="Carpeta destino donde crear subcarpetas por subclass",
    )
    p.add_argument(
        "--file-col",
        default="File",
        help="Nombre de la columna que contiene el nombre de archivo (por defecto: File)",
    )
    p.add_argument(
        "--subclass-col",
        default="Multiclass.Label",
        help="Nombre de la columna con la subclase (por defecto: Multiclass.Label)",
    )
    p.add_argument(
        "--sheet",
        default=None,
        help="Nombre o índice de la hoja en el Excel (opcional)",
    )
    p.add_argument(
        "--move", action="store_true", help="Mover archivos en vez de copiarlos"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    divide_images(
        excel_path=Path(args.excel),
        src_dir=Path(args.src),
        dest_dir=Path(args.dest),
        file_col=args.file_col,
        subclass_col=args.subclass_col,
        sheet_name=args.sheet,
        move=args.move,
    )
