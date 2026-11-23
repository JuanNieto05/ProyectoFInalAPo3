"""Carga del dataset a partir de data/splits.csv (fuente única de verdad de los splits).

Uso:
    from src.data.carga import cargar_splits, cargar_imagen
    df = cargar_splits()                 # todas las filas
    train = cargar_splits("train")       # solo un split
    img = cargar_imagen(train.iloc[0])   # PIL.Image RGB 224x224
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from src.config import PROCESSED_DIR, QUALITY_CLASSES, SPLITS_CSV


def cargar_splits(split: str | list[str] | None = None) -> pd.DataFrame:
    """Lee splits.csv y, opcionalmente, filtra por uno o varios splits.

    Añade la columna `y` con el índice de la clase de calidad según
    QUALITY_CLASSES (good=0, regular=1, bad=2).
    """
    df = pd.read_csv(SPLITS_CSV)
    if split is not None:
        splits = [split] if isinstance(split, str) else list(split)
        df = df[df["split"].isin(splits)].reset_index(drop=True)
    df["y"] = df["quality"].map({c: i for i, c in enumerate(QUALITY_CLASSES)})
    return df


def cargar_imagen(fila: pd.Series | dict) -> Image.Image:
    """Abre la imagen procesada de una fila de splits.csv como RGB."""
    return Image.open(PROCESSED_DIR / fila["path"]).convert("RGB")


def cargar_array(fila: pd.Series | dict) -> np.ndarray:
    """Imagen como arreglo float32 en [0, 1] con forma (alto, ancho, 3)."""
    return np.asarray(cargar_imagen(fila), dtype=np.float32) / 255.0
