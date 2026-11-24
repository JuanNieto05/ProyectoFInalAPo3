"""CNN pequeña entrenada desde cero para clasificar la calidad de la fruta.

Arquitectura (entrada 3×128×128):

    3 bloques [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]
    → promedio global por canal
    → Densa 64 + ReLU + Dropout → Densa 3 (logits de good / regular / bad)

Filtros: 16-32-64 en la red base (28.051 parámetros) y 32-64-128 en la principal,
`filtros_x2` (101.923). El promedio global sustituye al aplanado (128·16·16 = 32.768
entradas a la capa densa): mantiene la red pequeña, importante con ~3.900 imágenes y
solo 146 "regular".
"""
from __future__ import annotations

import torch
from torch import nn


class CNNCalidad(nn.Module):
    def __init__(self, filtros=(16, 32, 64), densa: int = 64, dropout: float = 0.3, n_clases: int = 3):
        super().__init__()
        capas, entrada = [], 3
        for f in filtros:
            capas += [nn.Conv2d(entrada, f, 3, padding=1, bias=False), nn.BatchNorm2d(f),
                      nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            entrada = f
        self.conv = nn.Sequential(*capas)
        self.clasificador = nn.Sequential(nn.Linear(entrada, densa), nn.ReLU(inplace=True),
                                          nn.Dropout(dropout), nn.Linear(densa, n_clases))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.conv(x).mean(dim=(2, 3)))


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
