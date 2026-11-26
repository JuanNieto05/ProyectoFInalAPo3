"""Segmentación fruta / fondo.

Dos niveles:

1. `mascara_fruta` (umbral simple): un píxel es "fruta" si su distancia euclídea RGB
   al color mediano del borde supera un umbral. Es el criterio de
   data/prepare_dataset.py para `area_ratio`. Rápido (~3 ms) pero ruidoso: incluye
   sombras, texturas del fondo y manos.

2. `segmentar_fruta` (GrabCut con priors, la que se usa en el proyecto): GrabCut
   [Rother et al., 2004] modela fondo y fruta con mezclas de gaussianas (GMM) de
   color y separa ambos con un corte mínimo en un grafo de píxeles. En lugar de un
   rectángulo, se inicializa con conocimiento previo:
     - "fondo seguro": franja del borde (3 %) cuyo color coincide con el fondo;
     - "fruta probable": píxeles distintos del fondo dentro de un círculo central
       (radio normalizado < 0,6), porque la fruta suele estar centrada;
     - "fruta segura": núcleo central (radio < 0,25) distinto del fondo y con
       saturación > 60 (la fruta tiene color; sombras y superficies grises no);
     - el resto, "fondo probable".
   Después se conserva solo el componente conexo más grande.
   ~200 ms por imagen de 224×224.

Limitaciones conocidas (ver docs/GUIA_SIGUIENTE_ETAPA.md):
  - En Kaggle la fruta suele estar en la mano; la mano puede quedar parcialmente en
    la máscara. Se probó excluir piel con un umbral en YCrCb, pero el tono de piel se
    solapa con manzanas, bananos y naranjas y borraba fruta: se descartó.
  - Si hay varias frutas, se conserva la región conexa más grande (puede ser un
    racimo o varias frutas que se tocan).
"""
from __future__ import annotations

import cv2
import numpy as np

UMBRAL_RGB = 40.0     # en escala 0-255, igual que prepare_dataset.py
SEMILLA_GRABCUT = 42  # GrabCut inicializa sus GMM con k-means aleatorio


def color_fondo(a: np.ndarray, frac_borde: float = 0.05) -> np.ndarray:
    """Mediana RGB de una franja de borde de ancho `frac_borde` del lado menor."""
    h, w, _ = a.shape
    b = max(2, int(frac_borde * min(h, w)))
    borde = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                            a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return np.median(borde, axis=0)


def mascara_fruta(a: np.ndarray, umbral: float = UMBRAL_RGB) -> np.ndarray:
    """Máscara simple (alto, ancho): píxeles cuyo color se aleja del fondo.

    `a` es la imagen RGB en escala 0-255 (uint8 o float).
    """
    a = a.astype(np.float32)
    return np.linalg.norm(a - color_fondo(a), axis=2) > umbral


def componente_mayor(m: np.ndarray) -> np.ndarray:
    """Conserva solo la región conexa (8-vecindad) de mayor área."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(bool)
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return etiquetas == k


def segmentar_fruta(rgb: np.ndarray, iteraciones: int = 5) -> np.ndarray:
    """Máscara booleana de la fruta con GrabCut inicializado con priors.

    `rgb` es la imagen RGB uint8 (alto, ancho, 3). Si GrabCut no puede
    ejecutarse (p. ej. no hay píxeles de fruta probables), devuelve la máscara
    simple.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    saturacion = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    distinto = mascara_fruta(rgb)

    yy, xx = np.mgrid[:h, :w]
    radio = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    b = max(2, int(0.03 * w))
    borde = np.zeros((h, w), bool)
    borde[:b] = borde[-b:] = True
    borde[:, :b] = borde[:, -b:] = True

    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[(radio < 0.6) & distinto] = cv2.GC_PR_FGD
    m[(radio < 0.25) & distinto & (saturacion > 60)] = cv2.GC_FGD
    m[borde & ~distinto] = cv2.GC_BGD

    if not np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).any():
        return componente_mayor(distinto)
    cv2.setRNGSeed(SEMILLA_GRABCUT)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, None,
                    np.zeros((1, 65)), np.zeros((1, 65)), iteraciones, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return componente_mayor(distinto)
    return componente_mayor((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD))


def aplicar_mascara(rgb: np.ndarray, mascara: np.ndarray, color=(128, 128, 128)) -> np.ndarray:
    """Reemplaza el fondo por un color neutro (para quitar el atajo del fondo)."""
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[~mascara] = color
    return out
