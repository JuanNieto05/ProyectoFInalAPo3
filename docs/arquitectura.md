# Arquitectura

## Pipeline
Imagen → segmentación (fondo) → [características manuales → ML clásico] / [imagen → CNN] → clase de calidad + tamaño → interfaz.

## Modelos

### SVM (ML clásico)
RBF con C = 30 y γ = 0,003 sobre 80 características de la fruta. Macro-F1 en `val`: 0,727.

### CNN desde cero (`src/models/cnn.py`)

| Etapa | Salida |
|---|---|
| Entrada: fruta sobre fondo gris, normalizada con train | 3×128×128 |
| Conv 3×3 (32) + BatchNorm + ReLU + MaxPool | 32×64×64 |
| Conv 3×3 (64) + BatchNorm + ReLU + MaxPool | 64×32×32 |
| Conv 3×3 (128) + BatchNorm + ReLU + MaxPool | 128×16×16 |
| Promedio global | 128 |
| Densa 64 + ReLU + Dropout 0,3 | 64 |
| Densa 3 | good / regular / bad |

101.923 parámetros. AdamW (lr 1e-3, weight decay 1e-4), lote 64, entropía cruzada con pesos de clase, aumento de
datos en train, reducción del lr (×0,5 tras 5 épocas sin mejora) y early stopping (paciencia 15, máximo 100 épocas)
por macro-F1 en `val`. Macro-F1 en `val`: 0,648 y 0,668 (dos semillas). Experimentos en
`notebooks/04_modeling_cnn.ipynb`.
