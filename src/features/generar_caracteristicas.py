"""Extrae las características manuales de todas las imágenes de data/splits.csv.

Genera dos tablas (una fila por imagen, columnas = filename + NOMBRES):
  data/features_fruta.csv     características solo dentro de la máscara (la que se usa)
  data/features_completa.csv  las mismas sobre la imagen completa, con fondo
                              (solo para el experimento de comparación del atajo)

Uso (desde la raíz del repo):
    python -m src.features.generar_caracteristicas
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config import DATA_DIR
from src.data.carga import cargar_imagen, cargar_mascara, cargar_splits
from src.features.extraccion import NOMBRES, extraer

SALIDAS = {"fruta": DATA_DIR / "features_fruta.csv", "completa": DATA_DIR / "features_completa.csv"}


def _una(fila):
    rgb = np.asarray(cargar_imagen(fila))
    m = cargar_mascara(fila)
    return extraer(rgb, m), extraer(rgb, np.ones_like(m))


def main():
    df = cargar_splits()
    t0 = time.time()
    res = Parallel(n_jobs=-1, batch_size=32)(delayed(_una)(f) for _, f in df.iterrows())
    for i, (nombre, ruta) in enumerate(SALIDAS.items()):
        tabla = pd.DataFrame(np.stack([r[i] for r in res]), columns=NOMBRES)
        tabla.insert(0, "filename", df["filename"].values)
        tabla.round(5).to_csv(ruta, index=False)
        print(f"{nombre}: {tabla.shape} -> {ruta}")
    print(f"Tiempo: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
