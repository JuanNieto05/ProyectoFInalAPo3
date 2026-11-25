"""Etiqueta manual de tamaño de las fotos propias (enunciado §2.3) y su comparación con d_eq.

    python -m src.evaluation.tamano_manual --plantilla   # crea data/tamano_manual.csv (no sobrescribe)
    python -m src.evaluation.tamano_manual               # compara, cuando la plantilla esté llena

Cómo llenar data/tamano_manual.csv (una fila por foto propia):
  tamano_manual  pequeno / mediano / grande, mirando la fruta real (o la foto si ya no se tiene),
                 comparada con otras frutas del MISMO tipo. Hacerlo SIN mirar las columnas
                 d_eq ni tamano_auto, para no sesgarse.
  diametro_cm    opcional: diámetro mayor de la fruta medido con regla o cinta, en cm.
  anotador       quién etiquetó (si etiquetan dos personas, se puede medir su acuerdo con kappa).

Qué calcula:
  - acuerdo entre la etiqueta manual y la automática (terciles de d_eq con cortes de train):
    exactitud y kappa ponderado lineal (las categorías son ordenadas: confundir pequeño con
    grande es peor que con mediano);
  - si hay diametro_cm: correlación de Spearman entre d_eq y el diámetro real. Si es baja,
    confirma que d_eq depende de la distancia de la cámara y no solo del tamaño.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import DATA_DIR, SIZE_CLASSES
from src.data.carga import cargar_splits

PLANTILLA = DATA_DIR / "tamano_manual.csv"


def crear_plantilla():
    if PLANTILLA.exists():
        print(f"Ya existe {PLANTILLA}; no se sobrescribe.")
        return
    df = cargar_splits()
    p = df[df.split == "test_propias"][["filename", "fruit", "quality", "d_eq", "tamano"]]
    p = p.rename(columns={"tamano": "tamano_auto"}).sort_values(["fruit", "filename"])
    p["tamano_manual"] = ""
    p["diametro_cm"] = ""
    p["anotador"] = ""
    p.to_csv(PLANTILLA, index=False)
    print(f"Plantilla creada: {PLANTILLA} ({len(p)} fotos). Llenar tamano_manual sin mirar d_eq ni tamano_auto.")


def comparar():
    t = pd.read_csv(PLANTILLA)
    llenas = t[t.tamano_manual.isin(SIZE_CLASSES)]
    if llenas.empty:
        print("La columna tamano_manual está vacía. Llénela con pequeno / mediano / grande.")
        return
    print(f"Fotos etiquetadas: {len(llenas)} de {len(t)}")
    orden = {c: i for i, c in enumerate(SIZE_CLASSES)}
    man, aut = llenas.tamano_manual.map(orden), llenas.tamano_auto.map(orden)
    print(f"Exactitud (manual = automática): {(man == aut).mean():.3f}")
    print(f"Kappa ponderado lineal: {cohen_kappa_score(man, aut, weights='linear'):.3f}")
    cm = pd.DataFrame(confusion_matrix(man, aut, labels=[0, 1, 2]),
                      index=[f"manual {c}" for c in SIZE_CLASSES], columns=[f"auto {c}" for c in SIZE_CLASSES])
    print(cm)
    print("\nPor fruta (exactitud):")
    print(llenas.assign(ok=(man == aut).values).groupby("fruit").ok.agg(["mean", "size"]).round(3))

    cm_col = pd.to_numeric(llenas.diametro_cm, errors="coerce")
    con_cm = llenas[cm_col.notna()]
    if len(con_cm) >= 5:
        rho, p = spearmanr(con_cm.d_eq, cm_col[cm_col.notna()])
        print(f"\nSpearman d_eq vs diámetro real (n={len(con_cm)}): rho = {rho:.3f} (p = {p:.3g})")
        for fruta, g in con_cm.groupby("fruit"):
            if len(g) >= 4:
                r, _ = spearmanr(g.d_eq, pd.to_numeric(g.diametro_cm))
                print(f"  {fruta}: rho = {r:.3f} (n={len(g)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantilla", action="store_true")
    a = ap.parse_args()
    crear_plantilla() if a.plantilla else comparar()
