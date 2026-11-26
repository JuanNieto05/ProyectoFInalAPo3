"""Predicción con los modelos finales a partir de una imagen (evaluación y app).

    rgb = preprocesar(Image.open(ruta))          # EXIF, RGB, cuadrado, 224x224
    mascara = segmentar_fruta(rgb)
    clase, prob = ModeloSVM().predecir(rgb, mascara)
    clase, prob = ModeloCNN().predecir(rgb, mascara)

`clase` es el índice en QUALITY_CLASSES y `prob` un vector de 3. En el SVM la clase sale
de predict() y NO siempre es el máximo de `prob` (Platt; ver GUIA_CNN_EVALUACION_APP.md).
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from data.prepare_dataset import pad_square_resize
from src.config import CHECKPOINTS_DIR, IMG_SIZE
from src.data.tensores import preparar
from src.features.extraccion import extraer
from src.training.entrenar_cnn import PRINCIPAL, cargar_checkpoint, predecir


def preprocesar(img: Image.Image) -> np.ndarray:
    """Mismo preprocesamiento que data/prepare_dataset.py. Devuelve RGB uint8 224x224."""
    return np.asarray(pad_square_resize(ImageOps.exif_transpose(img).convert("RGB"), IMG_SIZE[0]))


def diametro_equivalente(mascara: np.ndarray) -> float:
    """d_eq = 2*sqrt(area/pi), con el área como fracción de la imagen."""
    return float(2 * np.sqrt(mascara.mean() / np.pi))


class ModeloSVM:
    def __init__(self, archivo: str = "ml_mejor"):
        self.modelo = joblib.load(CHECKPOINTS_DIR / f"{archivo}.joblib")

    def predecir(self, rgb: np.ndarray, mascara: np.ndarray):
        x = extraer(rgb, mascara)[None]
        prob = self.modelo.predict_proba(x)[0] if hasattr(self.modelo, "predict_proba") else None
        return int(self.modelo.predict(x)[0]), prob


class ModeloCNN:
    def __init__(self, nombre: str = PRINCIPAL):
        self.modelo, self.media, self.desv, self.cfg = cargar_checkpoint(nombre)

    def predecir_lote(self, x: torch.Tensor):
        """x uint8 (N, 3, 128, 128) ya preparado. Devuelve (clases, probabilidades)."""
        prob = predecir(self.modelo, x, self.media, self.desv).softmax(1).numpy()
        return prob.argmax(1), prob

    def predecir(self, rgb: np.ndarray, mascara: np.ndarray):
        x = preparar(rgb, mascara if self.cfg["enmascarar"] else None)
        clase, prob = self.predecir_lote(torch.from_numpy(x.copy())[None])
        return int(clase[0]), prob[0]
