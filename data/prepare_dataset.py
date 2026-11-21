"""
prepare_dataset.py
------------------
Convierte las fotos propias (fotos_originales/) a un formato uniforme y,
opcionalmente, integra el dataset de Kaggle (Fruit Quality Classification)
en la MISMA estructura de carpetas.

Estructura de entrada (propias):
    fotos_originales/
        banana|manzana|naranja/
            good|regular|bad/  *.jpg|png|heic...

Estructura de salida (lista para torchvision ImageFolder / Keras
image_dataset_from_directory; la clase es la carpeta = calidad):
    data/processed/
        good/     banana_good_own_001.jpg, apple_good_kaggle_00123.jpg, ...
        regular/  ...
        bad/      ...
    data/manifest.csv   (un registro por imagen: fruta, calidad, fuente, tamano...)

Uso:
    python prepare_dataset.py --src fotos_originales
    python prepare_dataset.py --src fotos_originales --kaggle ruta/a/fruit-quality-classification

Dependencias: pillow, numpy (opcional: pillow-heif para fotos .heic de iPhone).
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # fotos HEIC de iPhone (opcional)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- Mapeo de nombres de carpeta -> nombre canonico (en ingles, como Kaggle) ---
FRUITS = {"banana": "banana", "manzana": "apple", "apple": "apple",
          "naranja": "orange", "orange": "orange"}
QUALITIES = {"good": "good", "bueno": "good", "buena": "good",
             "regular": "regular",
             "bad": "bad", "malo": "bad", "mala": "bad"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def load_image(path: Path) -> Image.Image:
    """Abre la imagen respetando la rotacion EXIF (fotos de celular) y en RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def estimate_size(img: Image.Image, thr: float = 40.0) -> tuple[float, float]:
    """
    Estima el tamano relativo de la fruta segmentando contra el fondo uniforme.
    Devuelve (area_ratio, diam_ratio):
      area_ratio = pixeles de fruta / pixeles totales
      diam_ratio = lado mayor del bounding box / lado mayor de la imagen
    Es valido solo si la distancia camara-fruta es parecida entre fotos.
    """
    small = img.copy()
    small.thumbnail((256, 256))
    a = np.asarray(small, dtype=np.float32)
    h, w, _ = a.shape
    b = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                             a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    mask = np.linalg.norm(a - bg, axis=2) > thr
    if mask.sum() == 0:
        return 0.0, 0.0
    ys, xs = np.where(mask)
    bbox_side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    return float(mask.mean()), float(bbox_side / max(h, w))


def pad_square_resize(img: Image.Image, size: int) -> Image.Image:
    """Hace la imagen cuadrada rellenando con el color de fondo (sin deformar)."""
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), bg)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def save_record(img, out_dir, fruit, quality, source, idx, orig_name, size, rows):
    name = f"{fruit}_{quality}_{source}_{idx:05d}.jpg"
    area, diam = estimate_size(img)
    pad_square_resize(img, size).save(out_dir / quality / name, quality=95)
    rows.append({"filename": name, "path": f"{quality}/{name}", "fruit": fruit,
                 "quality": quality, "source": source, "original": orig_name,
                 "area_ratio": round(area, 4), "diam_ratio": round(diam, 4)})


def process_own(src: Path, out_dir: Path, size: int, rows: list, seen=None, counters=None):
    """seen: conjunto (fruta, calidad, nombre_original) ya procesado (modo --append)."""
    counters = {} if counters is None else counters
    for fruit_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        fruit = FRUITS.get(fruit_dir.name.lower())
        if not fruit:
            print(f"[aviso] carpeta de fruta no reconocida, la omito: {fruit_dir.name}")
            continue
        for q_dir in sorted(p for p in fruit_dir.iterdir() if p.is_dir()):
            quality = QUALITIES.get(q_dir.name.lower())
            if not quality:
                print(f"[aviso] calidad no reconocida, la omito: {q_dir}")
                continue
            for f in sorted(q_dir.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                if seen is not None and (fruit, quality, f.name) in seen:
                    continue  # ya estaba en el manifest
                key = (fruit, quality)
                counters[key] = counters.get(key, 0) + 1
                try:
                    save_record(load_image(f), out_dir, fruit, quality, "own",
                                counters[key], f.name, size, rows)
                except Exception as e:  # archivo corrupto, HEIC sin soporte, etc.
                    print(f"[error] {f}: {e}")


def process_kaggle(kaggle: Path, out_dir: Path, size: int, rows: list):
    """
    Busca carpetas tipo 'Apple_Good', 'Banana_Bad', 'Orange_Good'... en cualquier
    nivel. Ignora frutas que no usamos (guava, lime, pomegranate) y la carpeta
    'Mixed Quality' (se segmenta aparte, ver README).
    """
    pat = re.compile(r"^(apple|banana|orange)[_ ]?(good|bad)$", re.I)
    counters = {}
    for d in sorted(p for p in kaggle.rglob("*") if p.is_dir()):
        m = pat.match(d.name)
        if not m or "mixed" in str(d).lower():
            continue
        fruit, quality = m.group(1).lower(), m.group(2).lower()
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            key = (fruit, quality)
            counters[key] = counters.get(key, 0) + 1
            try:
                save_record(load_image(f), out_dir, fruit, quality, "kaggle",
                            counters[key], f.name, size, rows)
            except Exception as e:
                print(f"[error] {f}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="fotos_originales", help="carpeta con fotos propias")
    ap.add_argument("--kaggle", default=None, help="carpeta descomprimida del dataset de Kaggle")
    ap.add_argument("--out", default="data", help="carpeta de salida")
    ap.add_argument("--size", type=int, default=224, help="lado final en pixeles")
    ap.add_argument("--clean", action="store_true", help="borra data/processed antes de empezar")
    ap.add_argument("--append", action="store_true",
                    help="agrega solo fotos propias NUEVAS sin tocar lo existente (manifest, regular de Kaggle)")
    args = ap.parse_args()
    if args.append and (args.clean or args.kaggle):
        raise SystemExit("--append no se combina con --clean ni --kaggle.")

    out = Path(args.out)
    proc = out / "processed"
    if args.clean and proc.exists():
        shutil.rmtree(proc)
    for q in ("good", "regular", "bad"):
        (proc / q).mkdir(parents=True, exist_ok=True)

    manifest = out / "manifest.csv"
    fields = ["filename", "path", "fruit", "quality", "source",
              "original", "area_ratio", "diam_ratio"]
    rows: list = []
    if args.append:
        # Modo incremental: parte del manifest actual y solo agrega fotos propias nuevas.
        with open(manifest, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            fields, rows = rd.fieldnames, list(rd)
        n_antes = len(rows)
        seen, counters = set(), {}
        for r in rows:
            if r["source"] != "own":
                continue
            # un archivo ya procesado puede haberse reetiquetado; se identifica por su original
            seen.add((r["fruit"], r["quality"], r["original"]))
            idx = int(r["filename"].rsplit("_", 1)[1].split(".")[0])
            k = (r["fruit"], r["quality"])
            counters[k] = max(counters.get(k, 0), idx)
        process_own(Path(args.src), proc, args.size, rows, seen, counters)
        print(f"\nFotos nuevas agregadas: {len(rows) - n_antes}")
    else:
        process_own(Path(args.src), proc, args.size, rows)
        if args.kaggle:
            process_kaggle(Path(args.kaggle), proc, args.size, rows)

    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nListo: {len(rows)} imagenes -> {proc}")
    counts = {}
    for r in rows:
        k = (r["source"], r["quality"])
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"  {k[0]:7s} {k[1]:8s} {counts[k]}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()