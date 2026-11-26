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


def macro_f1_prevalencia(y_true, y_pred, prevalencia=(1 / 3, 1 / 3, 1 / 3)) -> float:
    """Macro-F1 que tendría el modelo si las clases aparecieran con `prevalencia`.

    La precisión depende de la proporción de clases, y "regular" es 3,7 % en test y
    32,6 % en test_propias: el macro-F1 crudo de ambos no es comparable. Aquí se
    repondera cada fila de la matriz de confusión (los recalls no cambian).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(QUALITY_CLASSES)
    c = np.array([[((y_true == i) & (y_pred == j)).sum() for j in range(k)] for i in range(k)], float)
    c = c / np.maximum(c.sum(1, keepdims=True), 1) * np.asarray(prevalencia)[:, None]
    den = c.sum(0) + c.sum(1)
    return float(np.where(den > 0, 2 * np.diag(c) / np.where(den > 0, den, 1), 0).mean())


def bootstrap_diferencia(y_true, pred_a, pred_b, metrica, grupos=None, n: int = 1000,
                         semilla: int = 42, alfa: float = 0.05) -> tuple[float, float, float]:
    """(diferencia A - B, IC inferior, IC superior) con bootstrap PAREADO.

    Los dos modelos se evalúan en la misma remuestra: es la comparación correcta, porque dos
    intervalos individuales pueden solaparse y aun así diferir. Con `grupos` se remuestrean
    tomas completas (las fotos de una misma toma no son independientes).
    """
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    rng, vals = np.random.default_rng(semilla), []
    ids = np.arange(len(y)) if grupos is None else pd.factorize(np.asarray(grupos))[0]
    miembros = [np.where(ids == g)[0] for g in range(ids.max() + 1)]
    for _ in range(n):
        idx = np.concatenate([miembros[g] for g in rng.integers(0, len(miembros), len(miembros))])
        vals.append(metrica(y[idx], a[idx]) - metrica(y[idx], b[idx]))
    lo, hi = np.percentile(vals, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(metrica(y, a) - metrica(y, b)), float(lo), float(hi)
