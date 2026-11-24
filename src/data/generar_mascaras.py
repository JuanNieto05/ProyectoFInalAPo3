"""Calcula y guarda la máscara de la fruta de cada imagen de data/manifest.csv.

Las máscaras se guardan como PNG binarios (0 = fondo, 255 = fruta) en
data/masks/<misma ruta que en processed>.png, y un resumen por imagen en
data/masks.csv (área de la máscara, si se usó el respaldo, etc.).

Se calculan una sola vez (~20 min para 5.705 imágenes) y se reutilizan en la
extracción de características, en la CNN con fondo enmascarado y en el tamaño.

Uso (desde la raíz del repo):
    python -m src.data.generar_mascaras            # solo las que faltan
    python -m src.data.generar_mascaras --todas    # recalcula todo
"""
import argparse
import csv
import time

import numpy as np
from PIL import Image

import pandas as pd

from src.config import DATA_DIR, MANIFEST_CSV, MASKS_DIR
from src.data.carga import cargar_imagen
from src.data.segmentacion import segmentar_fruta

MASKS_CSV = DATA_DIR / "masks.csv"


def ruta_mascara(path_procesada: str):
    return MASKS_DIR / (path_procesada.rsplit(".", 1)[0] + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true", help="recalcula también las existentes")
    args = ap.parse_args()

    # se recorre manifest.csv (no splits.csv) porque make_splits.py usa estas
    # mascaras para calcular el tamano: deben existir antes de dividir
    df = pd.read_csv(MANIFEST_CSV)
    filas, t0 = [], time.time()
    for k, (_, f) in enumerate(df.iterrows(), start=1):
        destino = ruta_mascara(f["path"])
        if destino.exists() and not args.todas:
            m = np.asarray(Image.open(destino)) > 0
        else:
            m = segmentar_fruta(np.asarray(cargar_imagen(f)))
            destino.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m * 255).astype(np.uint8)).save(destino, optimize=True)
        ys, xs = np.where(m) if m.any() else (np.array([0]), np.array([0]))
        filas.append({"filename": f["filename"], "mask_area": round(float(m.mean()), 4),
                      "mask_bbox_lado": round(max(np.ptp(ys) + 1, np.ptp(xs) + 1) / m.shape[0], 4)})
        if k % 250 == 0:
            print(f"{k}/{len(df)}  ({time.time() - t0:.0f} s)", flush=True)

    with open(MASKS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Listo: {len(filas)} máscaras en {MASKS_DIR} y resumen en {MASKS_CSV}")


if __name__ == "__main__":
    main()
