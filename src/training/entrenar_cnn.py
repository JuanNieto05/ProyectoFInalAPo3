"""Entrenamiento de la CNN y de sus ablaciones (solo usa train y val).

    python -m src.training.entrenar_cnn            # todos los experimentos que falten
    python -m src.training.entrenar_cnn base       # solo los indicados

Cada experimento guarda:
    experiments/checkpoints/cnn_<nombre>.pt        mejor época (no versionado)
    experiments/results/cnn/<nombre>.csv           curvas por época
    experiments/results/cnn/<nombre>.json          configuración y métricas en val
Selección: early stopping por macro-F1 en val.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import CHECKPOINTS_DIR, RESULTS_DIR, SEED
from src.data.carga import cargar_splits
from src.data.tensores import aumentar, cargar_tensores, media_desv
from src.evaluation.metricas import resumen
from src.models.cnn import CNNCalidad, n_parametros

SALIDA = RESULTS_DIR / "cnn"
BASE = dict(enmascarar=True, aumento=True, balance="pesos", lr=1e-3, dropout=0.3, filtros=[16, 32, 64],
            wd=1e-4, lote=64, epocas=100, paciencia=15, semilla=SEED)
# Cada variante cambia UNA cosa respecto a BASE
EXPERIMENTOS = {
    "base": {},
    "lr_3e-3": {"lr": 3e-3},
    "sin_balance": {"balance": "ninguno"},
    "sin_aumento": {"aumento": False},
    "filtros_x2": {"filtros": [32, 64, 128]},
}


# CNN principal. La regla inicial (mayor macro-F1 en val) elegía "sin_aumento" (0,656) sobre
# "filtros_x2" (0,649), una diferencia que puede ser ruido (32 "regular" en val). Tras ver val, y sin
# mirar test ni test_propias, se eligió "filtros_x2": mejor F1 de "regular", menor brecha
# train-val y conserva el aumento, pensado para las fotos propias (más oscuras, fruta más pequeña).
PRINCIPAL = "filtros_x2"


def fijar_semillas(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def predecir(modelo, x, media, desv, lote=256) -> torch.Tensor:
    """Logits de imágenes uint8 ya preparadas."""
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(((b.float() / 255 - media) / desv).contiguous(memory_format=torch.channels_last)) for b in x.split(lote)])


def cargar_checkpoint(nombre: str):
    """(modelo, media, desv, cfg) de experiments/checkpoints/cnn_<nombre>.pt."""
    c = torch.load(CHECKPOINTS_DIR / f"cnn_{nombre}.pt", weights_only=True)
    m = CNNCalidad(filtros=c["cfg"]["filtros"], dropout=c["cfg"]["dropout"])
    m.load_state_dict(c["estado"])
    return m.eval(), c["media"], c["desv"], c["cfg"]


def entrenar(nombre: str, cambios: dict, datos: dict) -> dict:
    cfg = {**BASE, **cambios}
    fijar_semillas(cfg["semilla"])
    clave = cfg["enmascarar"]
    if clave not in datos:
        datos[clave] = [cargar_tensores(cargar_splits(s), clave) for s in ("train", "val")]
    (xtr, atr, ytr), (xva, _, yva) = datos[clave]
    media, desv = media_desv(xtr)

    n_c = torch.bincount(ytr, minlength=3).float()
    w = len(ytr) / (3 * n_c)                       # good 0,65, regular 8,87, bad 0,75
    pesos = {"pesos": w, "raiz": w.sqrt()}.get(cfg["balance"])
    perdida = nn.CrossEntropyLoss(weight=pesos)
    # channels_last: ~40 % más rápido en CPU, mismo resultado
    modelo = CNNCalidad(filtros=cfg["filtros"], dropout=cfg["dropout"]).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    print(f"[{nombre}] {n_parametros(modelo)} parámetros | {cfg}", flush=True)

    mejor, mejor_estado, mejor_ep, hist, t0 = -1.0, None, 0, [], time.time()
    for ep in range(1, cfg["epocas"] + 1):
        modelo.train()
        idx = (torch.multinomial(w[ytr], len(ytr), replacement=True) if cfg["balance"] == "sampler"
               else torch.randperm(len(ytr)))
        for b in idx.split(cfg["lote"]):
            x = xtr[b].float() / 255
            if cfg["aumento"]:
                x = aumentar(x, atr[b].float() / 255, "gris" if cfg["enmascarar"] else "reflejo")
            opt.zero_grad()
            loss = perdida(modelo(((x - media) / desv).contiguous(memory_format=torch.channels_last)), ytr[b])
            loss.backward()
            opt.step()
        # train y val se miden igual: modo eval, sin aumento, misma pérdida
        fila = {"epoca": ep, "lr": opt.param_groups[0]["lr"]}
        for s, x, y in (("train", xtr, ytr), ("val", xva, yva)):
            lg = predecir(modelo, x, media, desv)
            r = resumen(y.numpy(), lg.argmax(1).numpy())
            fila |= {f"loss_{s}": perdida(lg, y).item(), f"f1_{s}": r["macro-F1"],
                     f"f1_regular_{s}": r["F1 regular"]}
        hist.append(fila)
        plan.step(fila["f1_val"])
        if fila["f1_val"] > mejor:
            mejor, mejor_ep, mejor_estado = fila["f1_val"], ep, copy.deepcopy(modelo.state_dict())
        print(f"  {ep:2d} loss {fila['loss_train']:.3f}/{fila['loss_val']:.3f} "
              f"F1 {fila['f1_train']:.3f}/{fila['f1_val']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if ep - mejor_ep >= cfg["paciencia"]:
            break

    modelo.load_state_dict(mejor_estado)
    r = resumen(yva.numpy(), predecir(modelo, xva, media, desv).argmax(1).numpy())
    SALIDA.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": mejor_estado, "media": media, "desv": desv, "cfg": cfg},
               CHECKPOINTS_DIR / f"cnn_{nombre}.pt")
    pd.DataFrame(hist).round(5).to_csv(SALIDA / f"{nombre}.csv", index=False)
    out = {"nombre": nombre, "cfg": cfg, "mejor_epoca": mejor_ep, "epocas": len(hist),
           "segundos": round(time.time() - t0), "val": {k: round(float(v), 4) for k, v in r.items()}}
    (SALIDA / f"{nombre}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{nombre}] mejor época {mejor_ep}: macro-F1 val {mejor:.3f}", flush=True)
    return out


def resultado(nombre: str) -> dict:
    return json.loads((SALIDA / f"{nombre}.json").read_text(encoding="utf-8"))


def main(nombres: list[str]):
    datos: dict = {}

    def correr(n, cambios):
        if not ((SALIDA / f"{n}.json").exists() and (CHECKPOINTS_DIR / f"cnn_{n}.pt").exists()):
            entrenar(n, cambios, datos)

    if nombres:
        for n in nombres:
            correr(n, EXPERIMENTOS[n])
        return
    for n, cambios in EXPERIMENTOS.items():
        correr(n, cambios)
    # La configuración principal se repite con otra semilla (variabilidad) y, con la semilla
    # original, con imagen completa (experimento del atajo).
    segunda, completa = f"{PRINCIPAL}_s{SEED + 1}", f"{PRINCIPAL}_completa_s{SEED}"
    correr(segunda, {**EXPERIMENTOS[PRINCIPAL], "semilla": SEED + 1})
    correr(completa, {**EXPERIMENTOS[PRINCIPAL], "enmascarar": False})
    mayor = max(EXPERIMENTOS, key=lambda n: resultado(n)["val"]["macro-F1"])
    (SALIDA / "seleccion.json").write_text(json.dumps({
        "mejor": PRINCIPAL, "mayor_macro_f1_val": mayor, "criterio": "ver PRINCIPAL en entrenar_cnn.py",
        "semillas_fruta": [PRINCIPAL, segunda], "semillas_completa": [completa]}, indent=2), encoding="utf-8")
    print("CNN principal:", PRINCIPAL, "| mayor macro-F1 en val:", mayor)


if __name__ == "__main__":
    main(sys.argv[1:])
