"""Segmentación fruta / fondo para imágenes sobre fondo aproximadamente uniforme.

Es el mismo criterio que usa data/prepare_dataset.py (estimate_size) para calcular
`area_ratio`: el color de fondo se estima como la mediana de los píxeles del
borde, y un píxel es "fruta" si su distancia euclídea RGB a ese color supera un
umbral. Se usa en el EDA, en la extracción de características y en la app, para
que todos midan lo mismo.

Limitación conocida: falla si el fondo no es uniforme o si la fruta toca el
borde de la imagen (entonces el borde ya no representa al fondo).
"""
from __future__ import annotations

import numpy as np

UMBRAL_RGB = 40.0  # en escala 0-255, igual que prepare_dataset.py


def color_fondo(a: np.ndarray, frac_borde: float = 0.05) -> np.ndarray:
    """Mediana RGB de una franja de borde de ancho `frac_borde` del lado menor."""
    h, w, _ = a.shape
    b = max(2, int(frac_borde * min(h, w)))
    borde = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                            a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return np.median(borde, axis=0)


def mascara_fruta(a: np.ndarray, umbral: float = UMBRAL_RGB) -> np.ndarray:
    """Máscara booleana (alto, ancho) de los píxeles que no son fondo.

    `a` es la imagen RGB en escala 0-255 (uint8 o float).
    """
    a = a.astype(np.float32)
    return np.linalg.norm(a - color_fondo(a), axis=2) > umbral
