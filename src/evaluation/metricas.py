"""Métricas de evaluación del proyecto.

Métrica principal: macro-F1 = promedio simple del F1 de cada clase,
    F1_c = 2 * P_c * R_c / (P_c + R_c),  P_c = TP_c / (TP_c + FP_c),  R_c = TP_c / (TP_c + FN_c).
Al promediar sin ponderar, "regular" (3,8 % de los datos) pesa lo mismo que
"good" y "bad"; con accuracy, un modelo que nunca predice "regular" sacaría ~96 %.

Complementarias:
  - recall de "regular" y de "bad": costo de negocio (una fruta mala que pasa
    como buena llega al cliente; una regular no detectada no se vende con descuento).
  - accuracy balanceada = promedio de los recalls por clase.
  - kappa de Cohen: acuerdo con las etiquetas descontando el acuerdo por azar,
    kappa = (p_o - p_e) / (1 - p_e).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score, accuracy_score)

from src.config import QUALITY_CLASSES


def resumen(y_true, y_pred) -> dict:
    """Diccionario con las métricas del proyecto (clases 0=good, 1=regular, 2=bad)."""
    etiquetas = list(range(len(QUALITY_CLASSES)))
    f1 = f1_score(y_true, y_pred, labels=etiquetas, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=etiquetas, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=etiquetas, average=None, zero_division=0)
    out = {"macro-F1": f1.mean(),
           "acc. balanceada": balanced_accuracy_score(y_true, y_pred),
           "accuracy": accuracy_score(y_true, y_pred),
           "kappa": cohen_kappa_score(y_true, y_pred)}
    for i, c in enumerate(QUALITY_CLASSES):
        out[f"F1 {c}"] = f1[i]
        out[f"recall {c}"] = rec[i]
        out[f"precisión {c}"] = prec[i]
    return out


def tabla(resultados: dict[str, dict], columnas: list[str] | None = None) -> pd.DataFrame:
    """{nombre_modelo: resumen(...)} -> DataFrame redondeado."""
    df = pd.DataFrame(resultados).T
    if columnas:
        df = df[columnas]
    return df.astype(float).round(3)


def intervalo_bootstrap(y_true, y_pred, metrica=lambda a, b: f1_score(a, b, average="macro"),
                        n: int = 1000, semilla: int = 42, alfa: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza bootstrap (percentil) de una métrica.

    Útil en test_propias (46 imágenes), donde una sola cifra engaña: se remuestrea
    el conjunto con reemplazo n veces y se toman los percentiles alfa/2 y 1-alfa/2.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metrica(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 100 * alfa / 2)), float(np.percentile(vals, 100 * (1 - alfa / 2)))
