"""Imágenes como tensores para la CNN: preprocesamiento y aumento de datos.

`preparar` es el ÚNICO preprocesamiento de la CNN; la evaluación y la app deben usarlo
para que la entrada sea idéntica a la del entrenamiento:
    (con máscara) pintar el fondo de gris a 224×224 → redimensionar a 128×128 → uint8.
La normalización (media y desviación por canal, solo de train) se guarda en el checkpoint.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.carga import cargar_imagen, cargar_mascara
from src.data.segmentacion import aplicar_mascara

LADO = 128


def preparar(rgb: np.ndarray, mascara: np.ndarray | None, lado: int = LADO) -> np.ndarray:
    """RGB uint8 (H, W, 3) → uint8 (3, lado, lado). `mascara=None` conserva el fondo."""
    if mascara is not None:
        rgb = aplicar_mascara(rgb, mascara)
    return cv2.resize(rgb, (lado, lado), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)


def cargar_tensores(df: pd.DataFrame, enmascarar: bool = True):
    """(imágenes uint8 N×3×L×L, alfa uint8 N×1×L×L, etiquetas). alfa = 255 donde hay fruta."""
    xs, alfas = [], []
    for _, f in df.iterrows():
        rgb, m = np.array(cargar_imagen(f)), cargar_mascara(f)
        xs.append(preparar(rgb, m if enmascarar else None))
        a = m.astype(np.uint8) * 255 if enmascarar else np.full(m.shape, 255, np.uint8)
        alfas.append(cv2.resize(a, (LADO, LADO), interpolation=cv2.INTER_AREA)[None])
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(alfas)),
            torch.tensor(df["y"].to_numpy(copy=True), dtype=torch.long))


def media_desv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Media y desviación por canal en [0, 1]. Llamar solo con train."""
    v = x.float() / 255
    return v.mean((0, 2, 3)).view(1, 3, 1, 1), v.std((0, 2, 3)).view(1, 3, 1, 1)


def aumentar(x: torch.Tensor, alfa: torch.Tensor, relleno: str) -> torch.Tensor:
    """Aumento por lote sobre x en [0, 1].

    - Color, solo en la fruta: brillo ×[0,55, 1,25] (las fotos propias son más oscuras),
      contraste ×[0,8, 1,2] y saturación ×[0,8, 1,2]. No se cambia el tono: el color
      (verde, café) es parte de la definición de calidad.
    - Geometría: espejos, rotación ±30°, escala ×[0,6, 1,2] y traslación de hasta ±10 %
      (en las fotos propias la fruta ocupa menos imagen).
    `relleno`: "gris" (fruta segmentada) o "reflejo" (imagen completa).
    """
    n = len(x)

    def u(a, b):
        return torch.empty(n, 1, 1, 1).uniform_(a, b)

    c = x * u(0.55, 1.25)
    peso = alfa / alfa.sum((2, 3), keepdim=True).clamp(min=1)
    med = (c * peso).sum((1, 2, 3), keepdim=True) / 3
    c = med + (c - med) * u(0.8, 1.2)
    g = (c * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    c = (g + (c - g) * u(0.8, 1.2)).clamp(0, 1)
    x = x + alfa * (c - x)

    ang = u(-30, 30).flatten() * torch.pi / 180
    esc = u(0.6, 1.2).flatten()
    fx = torch.randint(0, 2, (n,)) * 2.0 - 1
    fy = torch.randint(0, 2, (n,)) * 2.0 - 1
    cos, sen = torch.cos(ang) / esc, torch.sin(ang) / esc
    # rotación · espejo: cada columna lleva su signo (no deforma la fruta)
    th = torch.stack([torch.stack([cos * fx, -sen * fy, u(-.2, .2).flatten()], 1),
                      torch.stack([sen * fx, cos * fy, u(-.2, .2).flatten()], 1)], 1)
    malla = F.affine_grid(th, list(x.shape), align_corners=False)
    if relleno == "reflejo":
        return F.grid_sample(x, malla, padding_mode="reflection", align_corners=False)
    gris = 128 / 255
    return F.grid_sample(x - gris, malla, padding_mode="zeros", align_corners=False) + gris
