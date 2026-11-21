"""
make_splits.py
--------------
Divide el dataset en entrenamiento / validacion / prueba y escribe data/splits.csv.


REGLAS:
  - Fotos de Kaggle: se reparten 70% train / 15% val / 15% test, por separado
    dentro de cada grupo (fruta, calidad). Asi cada split tiene la misma
    proporcion de clases ("estratificado"), incluso para "regular", que es chica.
  - Fotos propias (source == "own"): TODAS van al split "test_propias". Son la
    prueba externa: miden si el modelo generaliza a fotos reales distintas a Kaggle.
  - La semilla es fija (SEMILLA), asi que el resultado siempre es el mismo.
  - Tamano (pequeno/mediano/grande): se calcula con terciles de area_ratio de
    cada fruta, usando SOLO las fotos de train para fijar los cortes (si se
    calcularan con todo, el test influiria en las etiquetas).

Uso (desde la raiz del proyecto, despues de quitar duplicados):
    python data/make_splits.py
"""
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("data")
SEMILLA = 42
FRACCION_TRAIN = 0.70
FRACCION_VAL = 0.15   # el resto (0.15) es test


# ---------------------------------------------------------------- 1. Leer datos
with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
    filas = list(csv.DictReader(f))

for fila in filas:
    fila["split"] = ""

# ---------------------------------------------- 2. Fotos propias -> test externo
for fila in filas:
    if fila["source"] == "own":
        fila["split"] = "test_propias"

# ------------------------------------------ 3. Kaggle: dividir grupo por grupo
azar = random.Random(SEMILLA)
grupos = {}  # (fruta, calidad) -> lista de filas de Kaggle
for fila in filas:
    if fila["source"] == "kaggle":
        grupos.setdefault((fila["fruit"], fila["quality"]), []).append(fila)

for clave in sorted(grupos):
    miembros = sorted(grupos[clave], key=lambda r: r["filename"])  # orden fijo
    azar.shuffle(miembros)
    n = len(miembros)
    n_train = round(n * FRACCION_TRAIN)
    n_val = max(1, round(n * FRACCION_VAL))        # al menos 1 en val
    n_train = min(n_train, n - n_val - 1)          # y al menos 1 en test
    for k, fila in enumerate(miembros):
        if k < n_train:
            fila["split"] = "train"
        elif k < n_train + n_val:
            fila["split"] = "val"
        else:
            fila["split"] = "test"

# ----------------------- 4. Tamano por terciles de diam_ratio (cortes con train)
for fruta in sorted({r["fruit"] for r in filas}):
    area_train = [float(r["area_ratio"]) for r in filas
                  if r["fruit"] == fruta and r["split"] == "train"]
    corte_1, corte_2 = np.percentile(area_train, [100 / 3, 200 / 3])
    for r in filas:
        if r["fruit"] == fruta:
            d = float(r["area_ratio"])
            r["tamano"] = "pequeno" if d <= corte_1 else ("mediano" if d <= corte_2 else "grande")
    print(f"{fruta}: cortes de tamano (area_ratio) = {corte_1:.3f} / {corte_2:.3f}")

# ------------------------------------------------------------------ 5. Guardar
columnas = ["filename", "path", "fruit", "quality", "source", "split",
            "tamano", "area_ratio", "diam_ratio"]
with open(DATA / "splits.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
    w.writeheader()
    w.writerows(filas)

# ------------------------------------------------------------ 6. Comprobaciones
assert all(r["split"] for r in filas), "hay imagenes sin split"
assert len({r["filename"] for r in filas}) == len(filas), "nombres repetidos"

print(f"\nTotal: {len(filas)} imagenes -> {DATA / 'splits.csv'}\n")
orden = ["train", "val", "test", "test_propias"]
conteo = Counter((r["split"], r["quality"]) for r in filas)
print(f"{'split':14s}{'good':>8s}{'regular':>9s}{'bad':>8s}{'total':>8s}")
for s in orden:
    g, r_, b = (conteo[(s, q)] for q in ("good", "regular", "bad"))
    print(f"{s:14s}{g:8d}{r_:9d}{b:8d}{g + r_ + b:8d}")
print("\nTamano (todas las fotos):", dict(Counter(r["tamano"] for r in filas)))