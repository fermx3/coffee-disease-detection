from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from coffeedd.params import *

try:
    from PIL import Image, ImageOps
except Exception as e:
    raise RuntimeError("Pillow es requerido. Instala con: pip install pillow") from e


@dataclass
class ExportSummary:
    src_root: str
    dst_root: str
    total: int
    processed: int
    skipped: int
    errors: int
    policy: str
    target: int
    kept_format: bool
    dst_ext: Optional[str]


def _iter_images(
    root: Path, patterns: Sequence[str], recursive: bool = True
) -> Iterable[Path]:
    for pat in patterns:
        it = root.rglob(pat) if recursive else root.glob(pat)
        for p in it:
            if p.is_file():
                yield p


def _ensure_rgb(img: Image.Image, bg=(128, 128, 128)) -> Image.Image:
    # Orientación EXIF correcta
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGB":
        return img
    if img.mode in ("L", "I;16", "I"):
        return img.convert("RGB")
    if img.mode == "RGBA":
        # compositar fondo gris para alpha
        bg_img = Image.new("RGB", img.size, bg)
        bg_img.paste(img, mask=img.split()[-1])
        return bg_img
    return img.convert("RGB")


def _letterbox_square(
    img: Image.Image, target: int, pad_value: int = 128
) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target, target), (pad_value, pad_value, pad_value))
    scale = min(target / w, target / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target, target), (pad_value, pad_value, pad_value))
    left = (target - nw) // 2
    top = (target - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _center_crop_square(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((target, target), Image.Resampling.BICUBIC)


def _resize_stretch(img: Image.Image, target: int) -> Image.Image:
    return img.resize((target, target), Image.Resampling.BICUBIC)


def _dst_path_for(
    src_path: Path,
    src_root: Path,
    dst_root: Path,
    keep_format: bool,
    dst_ext: Optional[str],
) -> Path:
    rel = src_path.relative_to(src_root)
    if keep_format and dst_ext is None:
        # mantener la extensión original
        return (dst_root / rel).with_suffix(src_path.suffix.lower())
    # forzar extensión de salida
    ext = dst_ext if dst_ext is not None else ".jpg"
    return (dst_root / rel).with_suffix(ext.lower())


def export_preprocessed_images(
    src_root: Path | str,
    dst_root: Path | str,
    target: int = 224,
    policy: str = "letterbox",  # "letterbox" | "center-crop" | "resize"
    keep_format: bool = False,
    dst_ext: Optional[str] = ".jpg",  # ".jpg" | ".png" | None (si keep_format=True)
    quality: int = 95,
    recursive: bool = True,
    patterns: Sequence[str] = IMG_PATTERNS,
    clean_output: bool = False,
    workers: Optional[int] = None,
) -> ExportSummary:
    """
    Procesa imágenes físicamente y las guarda en disco preservando estructura de carpetas.

    - src_root: raíz de entrada (p.ej. data/raw_data o data/processed_data/splits/train).
    - dst_root: raíz de salida (p.ej. data/processed_data/resized_224 o .../splits_resized_224/train).
    - policy: letterbox (relleno para cuadrado), center-crop (recorte central), resize (estirar a cuadrado).
    - keep_format: si True, mantiene la extensión de cada imagen.
    - dst_ext: extensión de salida si no mantienes formato (por defecto .jpg).
    - quality: calidad JPEG si usas .jpg.
    - clean_output: si True, borra dst_root antes.
    - workers: hilos para paralelizar (por defecto 2x CPU lógico hasta 32).
    """
    src_root = Path(src_root).expanduser().resolve()
    dst_root = Path(dst_root).expanduser().resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Directorio de origen no válido: {src_root}")
    if src_root == dst_root or str(dst_root).startswith(str(src_root)):
        raise ValueError("dst_root no debe estar dentro de src_root.")

    if clean_output and dst_root.exists():
        # cuidado: borra todo el destino
        for p in dst_root.glob("*"):
            if p.is_dir():
                for root, dirs, files in os.walk(p, topdown=False):
                    for f in files:
                        Path(root, f).unlink(missing_ok=True)
                    for d in dirs:
                        Path(root, d).rmdir()
                p.rmdir()
            else:
                p.unlink()
    dst_root.mkdir(parents=True, exist_ok=True)

    paths = list(_iter_images(src_root, patterns, recursive=recursive))
    total = len(paths)
    if total == 0:
        return ExportSummary(
            str(src_root),
            str(dst_root),
            0,
            0,
            0,
            0,
            policy,
            target,
            keep_format,
            dst_ext,
        )

    def process_one(p: Path) -> Tuple[bool, Optional[str]]:
        try:
            with Image.open(p) as img0:
                img = _ensure_rgb(img0)
                if policy == "letterbox":
                    out = _letterbox_square(img, target)
                elif policy == "center-crop":
                    out = _center_crop_square(img, target)
                elif policy == "resize":
                    out = _resize_stretch(img, target)
                else:
                    return False, f"Política desconocida: {policy}"

                dst = _dst_path_for(p, src_root, dst_root, keep_format, dst_ext)
                dst.parent.mkdir(parents=True, exist_ok=True)
                save_kwargs = {}
                if dst.suffix.lower() in (".jpg", ".jpeg"):
                    save_kwargs = {"quality": quality, "optimize": True}
                out.save(dst, **save_kwargs)
            return True, None
        except Exception as e:
            return False, f"{p}: {e}"

    ok = 0
    err = 0
    workers = workers or min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, p): p for p in paths}
        for fut in as_completed(futs):
            success, _ = fut.result()
            if success:
                ok += 1
            else:
                err += 1

    return ExportSummary(
        src_root=str(src_root),
        dst_root=str(dst_root),
        total=total,
        processed=ok,
        skipped=total - ok - err,
        errors=err,
        policy=policy,
        target=target,
        kept_format=keep_format,
        dst_ext=dst_ext,
    )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Exportar imágenes preprocesadas a disco")
    ap.add_argument(
        "--src", required=True, help="Carpeta de entrada (p.ej. data/raw_data)"
    )
    ap.add_argument(
        "--dst",
        required=True,
        help="Carpeta de salida (p.ej. data/processed_data/resized_224)",
    )
    ap.add_argument("--target", type=int, default=224)
    ap.add_argument(
        "--policy", choices=["letterbox", "center-crop", "resize"], default="letterbox"
    )
    ap.add_argument(
        "--keep-format", action="store_true", help="Mantener extensión original"
    )
    ap.add_argument(
        "--dst-ext",
        default=".jpg",
        help="Forzar extensión de salida si no mantienes formato",
    )
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--clean-output", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    summary = export_preprocessed_images(
        src_root=args.src,
        dst_root=args.dst,
        target=args.target,
        policy=args.policy,
        keep_format=args.keep_format,
        dst_ext=(None if args.keep_format else args.dst_ext),
        quality=args.quality,
        clean_output=args.clean_output,
        workers=(None if args.workers <= 0 else args.workers),
    )
    print(json.dumps(summary.__dict__, indent=2, ensure_ascii=False))
