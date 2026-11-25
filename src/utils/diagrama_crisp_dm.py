"""Diagrama de flujo CRISP-DM personalizado del proyecto (docs/figures/crisp_dm_flujo.svg).

    python -m src.utils.diagrama_crisp_dm

Cada fase muestra lo que se hizo de verdad en el proyecto y el artefacto que produjo.
Las flechas punteadas son las vueltas atrás que ocurrieron (con su motivo): son la parte
"personalizada" del ciclo, lo que no se ve copiando el diagrama genérico.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.config import ROOT  # noqa: E402

SALIDA = ROOT / "docs" / "figures" / "crisp_dm_flujo.svg"

COLORES = ["#4C6EF5", "#15AABF", "#40C057", "#FAB005", "#FD7E14", "#E64980"]

FASES = [
    ("1. Comprensión del negocio",
     ["Clasificación manual: lenta,", "subjetiva, con desperdicio",
      "3 clases: good / regular / bad", "+ tamaño (terciles de d_eq)",
      "Meta: macro-F1 ≥ 0,85 (Kaggle)", "y ≥ 0,70 (fotos propias)"],
     "crisp_dm.md · criterios de calidad"),
    ("2. Comprensión de los datos",
     ["Kaggle (solo good/bad) + 46 fotos", "propias; regular = 3,9 %",
      "EDA: el FONDO delata la clase", "(F1 0,89 mirando solo el borde)",
      "Fotos propias: más oscuras,", "fruta más pequeña"],
     "notebook 02"),
    ("3. Preparación de los datos",
     ["EXIF, cuadrar, 224×224", "Duplicados (dHash) y conflictos",
      "Split 70/15/15 POR GRUPOS", "de toma (920 fugas → 0)",
      "GrabCut: fruta sobre gris", "80 características de la fruta"],
     "notebook 03 · splits.csv · masks/"),
    ("4. Modelado",
     ["5 modelos clásicos + GridSearch", "con StratifiedGroupKFold",
      "Pesos de clase (regular ×14)", "→ SVM RBF: 0,727 en val",
      "CNN desde cero + ablaciones", "y 2 semillas: 0,648 / 0,668"],
     "notebooks 04 · src/models"),
    ("5. Evaluación",
     ["Una sola vez en test y propias", "SVM 0,723 (test) / 0,470 (propias)",
      "IC bootstrap por tomas", "Atajo del fondo, brecha de dominio",
      "Errores con imágenes:", "sombra, regular, criterio verde"],
     "notebook 05"),
    ("6. Despliegue",
     ["App Streamlit: cámara o archivo", "Mismo preprocesamiento",
      "Máscara visible al usuario", "Aviso de resultado dudoso",
      "Tamaño con distancia fija", "Uso como APOYO, no decisión"],
     "app/app.py"),
]

# (desde, hasta, texto, curvatura): vueltas atrás reales del proyecto
VUELTAS = [
    (1, 0, "Kaggle no trae «regular»:\ndefinir criterio y reetiquetar 210", -0.45),
    (2, 1, "Fuga por fotos casi iguales\nentre splits → volver a revisar", -0.45),
    (3, 2, "Características de la\nimagen completa usan\nel fondo → sobre la\nfruta; area_ratio\nfalla → d_eq", 0.0),
    (4, 2, "Sombras y relleno rompen la máscara\n→ corregir segmentación (trabajo futuro)", 0.32),
    (4, 0, "Metas no alcanzadas en fotos\npropias → alcance: apoyo, no reemplazo", 0.42),
    (5, 3, "predict ≠ predict_proba\n→ mostrar puntaje y «dudoso»", 0.55),
]


def main():
    fig, ax = plt.subplots(figsize=(17.2, 10.0))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 10.0)
    ax.axis("off")

    ancho, alto = 4.4, 2.95
    xs = [0.55, 5.8, 11.05]
    ys = [5.35, 1.05]
    # fila 1: 1 → 2 → 3 ; fila 2 (de derecha a izquierda): 4 ← ... para que el flujo sea una "S"
    posiciones = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[2], ys[0]),
                  (xs[2], ys[1]), (xs[1], ys[1]), (xs[0], ys[1])]
    centros = []
    for i, ((titulo, lineas, artefacto), (x, y)) in enumerate(zip(FASES, posiciones)):
        c = COLORES[i]
        ax.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc="white", ec=c, lw=2.2))
        ax.add_patch(FancyBboxPatch((x, y + alto - 0.52), ancho, 0.52,
                                    boxstyle="round,pad=0.02,rounding_size=0.18", fc=c, ec=c, lw=2.2))
        ax.text(x + ancho / 2, y + alto - 0.26, titulo, ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold")
        for k, linea in enumerate(lineas):
            ax.text(x + 0.18, y + alto - 0.8 - k * 0.3, linea, ha="left", va="center", fontsize=10.2, color="#212529")
        ax.text(x + ancho - 0.12, y + 0.14, artefacto, ha="right", va="bottom", fontsize=8.8,
                color=c, style="italic", fontweight="bold")
        centros.append((x, y))

    def flecha(p, q, **kw):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, **kw))

    # flujo principal (sólido)
    flecha((xs[0] + ancho, ys[0] + alto / 2), (xs[1], ys[0] + alto / 2), color="#343a40", lw=2.4)
    flecha((xs[1] + ancho, ys[0] + alto / 2), (xs[2], ys[0] + alto / 2), color="#343a40", lw=2.4)
    flecha((xs[2] + ancho / 2, ys[0]), (xs[2] + ancho / 2, ys[1] + alto), color="#343a40", lw=2.4)
    flecha((xs[2], ys[1] + alto / 2), (xs[1] + ancho, ys[1] + alto / 2), color="#343a40", lw=2.4)
    flecha((xs[1], ys[1] + alto / 2), (xs[0] + ancho, ys[1] + alto / 2), color="#343a40", lw=2.4)

    # vueltas atrás (punteadas)
    def ancla(i, lado):
        x, y = centros[i]
        return {"arriba": (x + ancho / 2, y + alto), "abajo": (x + ancho / 2, y),
                "izq": (x, y + alto / 2), "der": (x + ancho, y + alto / 2)}[lado]

    gris = "#868e96"
    # 2 → 1 y 3 → 2: por encima de la fila superior
    for (desde, hasta, texto, _), dx in ((VUELTAS[0], 0.0), (VUELTAS[1], 0.0)):
        p, q = ancla(desde, "arriba"), ancla(hasta, "arriba")
        p, q = (p[0] - 0.9, p[1] + 0.05), (q[0] + 0.9, q[1] + 0.05)
        flecha(p, q, color=gris, lw=1.6, ls="--", connectionstyle="arc3,rad=0.3")
        ax.text((p[0] + q[0]) / 2, p[1] + 0.92, texto, ha="center", va="center", fontsize=9.2, color="#495057")
    # 4 → 3 (Modelado → Preparación): por fuera, a la derecha
    p, q = (xs[2] + ancho, ys[1] + alto * 0.75), (xs[2] + ancho, ys[0] + alto * 0.25)
    flecha(p, q, color=gris, lw=1.6, ls="--", connectionstyle="arc3,rad=0.5")
    ax.text(14.45, 4.68, VUELTAS[2][2], ha="center", va="center", fontsize=9.2, color="#495057")
    # 5 → 3 (Evaluación → Preparación): entre filas, lado derecho
    p, q = (xs[1] + ancho * 0.78, ys[1] + alto), (xs[2] + 0.35, ys[0])
    flecha(p, q, color=gris, lw=1.6, ls="--", connectionstyle="arc3,rad=-0.25")
    ax.text(10.6, 4.72, VUELTAS[3][2], ha="center", va="center", fontsize=9.2, color="#495057",
            bbox=dict(fc="white", ec="none", pad=1.5))
    # 5 → 1 (Evaluación → Negocio): entre filas, lado izquierdo
    p, q = (xs[1] + ancho * 0.22, ys[1] + alto), (xs[0] + ancho * 0.7, ys[0])
    flecha(p, q, color=gris, lw=1.6, ls="--", connectionstyle="arc3,rad=0.25")
    ax.text(5.4, 4.72, VUELTAS[4][2], ha="center", va="center", fontsize=9.2, color="#495057",
            bbox=dict(fc="white", ec="none", pad=1.5))
    # 6 → 4 (Despliegue → Modelado): por debajo de la fila inferior
    p, q = (xs[0] + ancho / 2 + 0.9, ys[1] - 0.05), (xs[2] + ancho / 2 - 0.9, ys[1] - 0.05)
    flecha(p, q, color=gris, lw=1.6, ls="--", connectionstyle="arc3,rad=0.12")
    ax.text(8.9, 0.15, VUELTAS[5][2].replace("\n", " "), ha="center", va="center", fontsize=9.2, color="#495057")

    ax.text(8.0, 9.8, "CRISP-DM aplicado: clasificación de calidad y tamaño de fruta", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#212529")
    ax.text(0.55, 0.15, "——  flujo principal      - - -  vuelta atrás (con su motivo)", ha="left", va="center",
            fontsize=9, color="#495057")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, bbox_inches="tight")
    fig.savefig(SALIDA.with_suffix(".png"), dpi=110, bbox_inches="tight")
    print("Guardado:", SALIDA)


if __name__ == "__main__":
    main()
