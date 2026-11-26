"""Características manuales de la fruta para los modelos de ML clásico.

Todas se calculan SOLO sobre los píxeles de la máscara (src/data/segmentacion.py),
para que el modelo no use el fondo (atajo detectado en el EDA). No se incluyen el
tipo de fruta ni el tamaño: la calidad no debe depender de qué tan cerca se tomó
la foto.

Grupos de características (80 en total):

  color     Estadísticos (media, desviación, percentiles 10/50/90) de cada canal
            en HSV y CIELAB. Lab separa luminosidad (L*) de color (a*: verde-rojo,
            b*: azul-amarillo) y es aproximadamente uniforme perceptualmente: una
            distancia en Lab se parece a la diferencia que percibe una persona.
            El tono H es circular (0° = 360°), por eso se resume con la media
            circular (cos, sin) y la longitud del vector medio.
  hist      Histograma normalizado del tono (12 bins) y de la saturación y el
            brillo (8 bins cada uno): describe la proporción de cada color.
  manchas   Proporción de píxeles "anómalos" respecto a la propia fruta: más
            oscuros que la mediana de L* de la fruta en > 15/25/35 unidades, o con
            un color (a*, b*) a más de 15/25 unidades de su color mediano. Mide
            golpes y podredumbre de forma relativa a cada fruta, así que no
            depende del tipo de fruta ni de la iluminación global. También el
            número y el área del mayor "blob" oscuro.
  textura   LBP uniforme (P=8, R=1; 10 bins) y GLCM (contraste, homogeneidad,
            energía, correlación, promedio de 4 direcciones) en gris dentro de la
            máscara, y densidad de bordes de Canny. La piel sana es lisa; la
            arrugada, golpeada o con moho tiene más textura.
  forma     Solidez (área / área de la envolvente convexa) y excentricidad de la
            máscara: las frutas deformadas o aplastadas tienen menor solidez.

Uso:
    from src.features.extraccion import extraer, NOMBRES
    x = extraer(rgb_uint8, mascara_bool)        # vector de len(NOMBRES)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import label, regionprops

PERCENTILES = (10, 50, 90)
UMBRALES_OSCURO = (15, 25, 35)   # unidades de L* (0-100) por debajo de la mediana
UMBRALES_COLOR = (15, 25)        # distancia en el plano (a*, b*)
BINS_H, BINS_S, BINS_V = 12, 8, 8
LBP_P, LBP_R = 8, 1


def _nombres() -> list[str]:
    n = []
    for esp, canales in (("hsv", ("S", "V")), ("lab", ("L", "a", "b"))):
        for c in canales:
            n += [f"{esp}_{c}_media", f"{esp}_{c}_std"] + [f"{esp}_{c}_p{p}" for p in PERCENTILES]
    n += ["hsv_H_cos", "hsv_H_sin", "hsv_H_concentracion"]
    n += [f"hist_H_{i}" for i in range(BINS_H)] + [f"hist_S_{i}" for i in range(BINS_S)] \
        + [f"hist_V_{i}" for i in range(BINS_V)]
    n += [f"mancha_oscura_{u}" for u in UMBRALES_OSCURO] + [f"mancha_color_{u}" for u in UMBRALES_COLOR]
    n += ["mancha_n_blobs", "mancha_blob_mayor"]
    n += [f"lbp_{i}" for i in range(LBP_P + 2)]
    n += ["glcm_contraste", "glcm_homogeneidad", "glcm_energia", "glcm_correlacion", "bordes_densidad"]
    n += ["forma_solidez", "forma_excentricidad"]
    return n


NOMBRES = _nombres()


def _stats(v: np.ndarray) -> list[float]:
    return [float(v.mean()), float(v.std())] + [float(x) for x in np.percentile(v, PERCENTILES)]


def extraer(rgb: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Vector de características de la fruta. `rgb` uint8 (H, W, 3); `mascara` bool (H, W)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    m = mascara.astype(bool)
    if m.sum() < 30:                      # máscara vacía: se usa toda la imagen
        m = np.ones(m.shape, bool)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)        # H 0-179, S y V 0-255
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # L 0-100, a y b ~ -128..127
    H, S, V = hsv[..., 0][m] * 2.0, hsv[..., 1][m] / 255.0, hsv[..., 2][m] / 255.0
    L, A, B = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]

    f: list[float] = []
    # --- color
    for v in (S, V, L, A, B):
        f += _stats(v)
    ang = np.radians(H)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    f += [float(c), float(s), float(np.hypot(c, s))]
    # --- histogramas
    f += list(np.histogram(H, bins=BINS_H, range=(0, 360))[0] / H.size)
    f += list(np.histogram(S, bins=BINS_S, range=(0, 1))[0] / S.size)
    f += list(np.histogram(V, bins=BINS_V, range=(0, 1))[0] / V.size)
    # --- manchas (relativas a la propia fruta)
    l_med, a_med, b_med = np.median(L), np.median(A), np.median(B)
    f += [float((L < l_med - u).mean()) for u in UMBRALES_OSCURO]
    dist_color = np.hypot(A - a_med, B - b_med)
    f += [float((dist_color > u).mean()) for u in UMBRALES_COLOR]
    oscuro = np.zeros(m.shape, bool)
    oscuro[m] = L < l_med - UMBRALES_OSCURO[1]
    oscuro = cv2.morphologyEx(oscuro.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    etiquetas = label(oscuro)
    areas = np.bincount(etiquetas.ravel())[1:] if etiquetas.max() else np.array([0])
    f += [float((areas >= 10).sum()), float(areas.max() / m.sum())]
    # --- textura
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gris, LBP_P, LBP_R, method="uniform")
    f += list(np.histogram(lbp[m], bins=LBP_P + 2, range=(0, LBP_P + 2))[0] / m.sum())
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    recorte = (gris[y0:y1, x0:x1] // 8).astype(np.uint8)          # 32 niveles de gris
    recorte[~m[y0:y1, x0:x1]] = 0                                   # nivel 0 = fuera de la fruta
    glcm = graycomatrix(recorte, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                        symmetric=True, normed=False).astype(np.float64)
    glcm[0, :] = 0
    glcm[:, 0] = 0                                                  # ignora pares con el fondo
    glcm /= max(glcm.sum(axis=(0, 1), keepdims=True).min(), 1)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        f.append(float(np.nan_to_num(graycoprops(glcm, prop)).mean()))
    bordes = cv2.Canny(gris, 50, 150) > 0
    interior = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # sin el contorno
    f.append(float(bordes[interior].mean()) if interior.any() else 0.0)
    # --- forma
    props = regionprops(m.astype(np.uint8))[0]
    f += [float(props.solidity), float(props.eccentricity)]

    out = np.array(f, dtype=np.float32)
    assert out.size == len(NOMBRES), (out.size, len(NOMBRES))
    return out
