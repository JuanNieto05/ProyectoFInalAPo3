"""
find_duplicates.py
-------------------------
encontrar fotos duplicadas en data/processed.

  1. Convertimos cada foto en una "huella" de 256 valores True/False (dHash).
     Dos fotos casi iguales tienen huellas casi iguales, aunque se hayan
     reescalado o recomprimido.
  2. Comparamos cada foto con todas las demas y contamos en cuantos de los 256
     valores difieren (distancia de Hamming). Si difieren en pocos (<= UMBRAL),
     las consideramos duplicadas.
  3. Tambien comparamos contra la version ESPEJADA, porque girar una foto de
     izquierda a derecha es una "augmentacion" muy comun.

Uso (desde la raiz del proyecto):
    python data/find_duplicates.py           -> solo reporta
    python data/find_duplicates.py --mover   -> mueve los duplicados
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DATA = Path("data")
PROCESSED = DATA / "processed"
UMBRAL = 10  # maximo de diferencias (de 256) para llamarlas "duplicadas"


def calcular_huella(imagen):
    """Devuelve 256 valores True/False que resumen la imagen (dHash).

    - Pasamos a blanco y negro y la achicamos a 17 columnas x 16 filas.
    - Para cada pixel preguntamos: es mas claro que el pixel de su izquierda?
      17 columnas dan 16 comparaciones por fila; 16 filas x 16 = 256 respuestas.
    """
    gris = imagen.convert("L").resize((17, 16))
    px = np.asarray(gris, dtype=int)
    return (px[:, 1:] > px[:, :-1]).flatten()


# ---------------------------------------------------------------- 1. Leer datos
with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    campos = lector.fieldnames
    filas = list(lector)

print(f"Calculando huellas de {len(filas)} imagenes...")
huellas = []           # huella de cada imagen
huellas_espejo = []    # huella de cada imagen volteada de izquierda a derecha
for fila in filas:
    img = Image.open(PROCESSED / fila["path"])
    huellas.append(calcular_huella(img))
    huellas_espejo.append(calcular_huella(ImageOps.mirror(img)))
huellas = np.array(huellas)                # tabla de N filas x 256 columnas
huellas_espejo = np.array(huellas_espejo)

# ---------------------------------------------------------- 2. Comparar parejas

pares = []  # (i, j, distancia)
for i in range(len(filas) - 1):
   
    dist_normal = (huellas[i] != huellas[i + 1:]).sum(axis=1)
    dist_espejo = (huellas[i] != huellas_espejo[i + 1:]).sum(axis=1)
    dist = np.minimum(dist_normal, dist_espejo)   # nos quedamos con la menor
    for k in np.where(dist <= UMBRAL)[0]:
        pares.append((i, i + 1 + k, int(dist[k])))

print(f"Parejas duplicadas encontradas: {len(pares)}")

# ------------------------------------------------------------ 3. Guardar listado
with open(DATA / "duplicates.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["foto_a", "foto_b", "distancia", "misma_etiqueta"])
    for i, j, d in sorted(pares, key=lambda p: p[2]):
        a, b = filas[i], filas[j]
        misma = a["fruit"] == b["fruit"] and a["quality"] == b["quality"]
        w.writerow([a["filename"], b["filename"], d, int(misma)])

# Vista previa: las 20 parejas MAS dudosas (mayor distancia), una al lado de otra,
# para comprobar a ojo si el UMBRAL esta bien puesto.
carpeta = DATA / "duplicates_preview"
if carpeta.exists():
    shutil.rmtree(carpeta)
carpeta.mkdir()
for n, (i, j, d) in enumerate(sorted(pares, key=lambda p: -p[2])[:20], start=1):
    a = Image.open(PROCESSED / filas[i]["path"]).convert("RGB")
    b = Image.open(PROCESSED / filas[j]["path"]).convert("RGB")
    lienzo = Image.new("RGB", (a.width + b.width + 10, a.height), "white")
    lienzo.paste(a, (0, 0))
    lienzo.paste(b, (a.width + 10, 0))
    lienzo.save(carpeta / f"pareja_{n:02d}_dist{d}.jpg")

# ----------------------------------------------------------- 4. Mover (opcional)
if "--mover" in sys.argv:
    destino = DATA / "duplicates_removed"
    destino.mkdir(exist_ok=True)
    movidas = set()
    for i, j, d in pares:
        a, b = filas[i], filas[j]
        if a["fruit"] != b["fruit"] or a["quality"] != b["quality"]:
            continue  # etiquetas distintas: se decide a mano, no se toca
        if i in movidas or j in movidas:
            continue
        # si una es propia y la otra de Kaggle, nos quedamos con la propia
        sobra = i if (a["source"] != "own" and b["source"] == "own") else j
        fila = filas[sobra]
        shutil.move(str(PROCESSED / fila["path"]), str(destino / fila["filename"]))
        movidas.add(sobra)
    restantes = [fila for k, fila in enumerate(filas) if k not in movidas]
    with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(restantes)
    print(f"Movidas {len(movidas)} fotos a {destino}. Quedan {len(restantes)} en el manifest.")
else:
    print("Revisa data/duplicates.csv y data/duplicates_preview/.")
    print("Si todo se ve bien, ejecuta de nuevo con --mover.")