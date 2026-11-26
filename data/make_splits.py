"""
make_splits.py
--------------
Divide el dataset en entrenamiento / validacion / prueba y escribe data/splits.csv.


REGLAS:
  - Fotos de Kaggle: se reparten 70% train / 15% val / 15% test, por separado
    dentro de cada estrato (fruta, calidad). Asi cada split tiene la misma
    proporcion de clases ("estratificado"), incluso para "regular", que es chica.
  - Split POR GRUPOS: Kaggle trae varias tomas casi iguales de la misma fruta
    (mismo encuadre, leves cambios de color/recorte). Si una queda en train y su
    gemela en test, el test mide memoria y no generalizacion (fuga de datos).
    Por eso se agrupan (union-find) las fotos de la misma fruta si cumplen
    ALGUNA de estas condiciones, y cada grupo entero va a UN solo split:
      a) Casi identicas: huella dHash a <= UMBRAL_GRUPO bits (de 256, comparando
         tambien con la version espejada; misma huella que find_duplicates.py).
      b) Misma rafaga: el nombre original de Kaggle trae la hora de captura
         (IMG20200729173755, IMG_20190910_171724, 20190809_164444) o el contador
         de la camara (IMG_9363). Fotos consecutivas tomadas con <= RAFAGA_SEG
         segundos (o <= RAFAGA_CONTADOR numeros) de diferencia y con dHash
         <= UMBRAL_RAFAGA son la misma fruta en la misma toma.
         (Medido: con <= 1 s la mediana de dHash es 33; desde 3 s sube a ~104,
         casi como dos fotos al azar ~128. El tope de dHash evita que una sesion
         continua de fotos encadene cientos de frutas distintas en un grupo.)
    Un grupo que contiene alguna "regular" se estratifica como "regular" para
    que esa clase quede bien repartida. La columna `grupo` queda en splits.csv.
  - Fotos propias (source == "own"): TODAS van al split "test_propias". Son la
    prueba externa: miden si el modelo generaliza a fotos reales distintas a Kaggle.
  - La semilla es fija (SEMILLA), asi que el resultado siempre es el mismo.
  - Tamano (pequeno/mediano/grande): terciles por fruta del diametro equivalente
    normalizado d_eq = 2*sqrt(mask_area/pi), donde mask_area es la fraccion de la
    imagen cubierta por la mascara GrabCut (data/masks.csv, generado con
    `python -m src.data.generar_mascaras`). Los cortes se fijan SOLO con train
    (si se calcularan con todo, el test influiria en las etiquetas).
    Si data/masks.csv no existe se usa area_ratio (umbral simple), que es poco
    fiable en fondos con textura (ver notebooks/03_data_preparation.ipynb).

Uso (desde la raiz del proyecto, despues de quitar duplicados):
    python data/make_splits.py
"""
import csv
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DATA = Path("data")
PROCESSED = DATA / "processed"
SEMILLA = 42
FRACCION_TRAIN = 0.70
FRACCION_VAL = 0.15   # el resto (0.15) es test
UMBRAL_GRUPO = 20     # bits distintos (de 256) para considerar dos fotos "la misma toma"
RAFAGA_SEG = 2        # segundos maximos entre fotos consecutivas de una rafaga
RAFAGA_CONTADOR = 5   # diferencia maxima en el contador de camara (IMG_9363 -> IMG_9368)
UMBRAL_RAFAGA = 60    # dHash maximo para unir dos fotos consecutivas de una rafaga


def calcular_huella(imagen):
    """dHash de 256 bits (igual que en find_duplicates.py)."""
    gris = imagen.convert("L").resize((17, 16))
    px = np.asarray(gris, dtype=int)
    return (px[:, 1:] > px[:, :-1]).flatten()


def momento_captura(nombre_original):
    """Lee del nombre original de Kaggle la hora de captura o el contador de camara.

    Devuelve (tipo, valor): ("t", segundos) | ("c", contador) | (None, None).
    """
    t = re.search(r"(20\d{6})_?(\d{6})", nombre_original)
    if t:
        hora = datetime.strptime(t.group(1) + t.group(2), "%Y%m%d%H%M%S")
        return "t", hora.timestamp()
    c = re.search(r"IMG_(\d{4})\.", nombre_original, re.I)
    if c:
        return "c", int(c.group(1))
    return None, None


def agrupar_similares(filas):
    """Agrupa (union-find) las fotos que son la misma fruta en la misma toma.

    Une dos fotos de la misma fruta si son casi identicas (dHash <= UMBRAL_GRUPO)
    o si son consecutivas en una rafaga de Kaggle (ver docstring del modulo).
    Devuelve una lista con el id de grupo de cada fila.
    """
    huellas, espejos = [], []
    for fila in filas:
        img = Image.open(PROCESSED / fila["path"])
        huellas.append(calcular_huella(img))
        espejos.append(calcular_huella(ImageOps.mirror(img)))
    huellas, espejos = np.array(huellas), np.array(espejos)

    padre = list(range(len(filas)))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    frutas = np.array([f["fruit"] for f in filas])
    for i in range(len(filas) - 1):
        d = np.minimum((huellas[i] != huellas[i + 1:]).sum(axis=1),
                       (huellas[i] != espejos[i + 1:]).sum(axis=1))
        for k in np.where((d <= UMBRAL_GRUPO) & (frutas[i + 1:] == frutas[i]))[0]:
            padre[raiz(i)] = raiz(i + 1 + k)

    # Rafagas: se ordenan por momento de captura dentro de cada carpeta original de
    # Kaggle (fruta + calidad original, antes de reetiquetar a "regular").
    secuencias = {}
    for i, fila in enumerate(filas):
        if fila["source"] != "kaggle":
            continue
        tipo, valor = momento_captura(fila["original"])
        if tipo is None:
            continue
        calidad_original = fila.get("relabeled_from") or fila["quality"]
        secuencias.setdefault((fila["fruit"], calidad_original, tipo), []).append((valor, i))
    for (_, _, tipo), lista in secuencias.items():
        limite = RAFAGA_SEG if tipo == "t" else RAFAGA_CONTADOR
        lista.sort()
        for (v1, i), (v2, j) in zip(lista, lista[1:]):
            dist = min((huellas[i] != huellas[j]).sum(), (huellas[i] != espejos[j]).sum())
            if v2 - v1 <= limite and dist <= UMBRAL_RAFAGA:
                padre[raiz(i)] = raiz(j)
    return [raiz(i) for i in range(len(filas))]


# ---------------------------------------------------------------- 1. Leer datos
with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
    filas = list(csv.DictReader(f))

for fila in filas:
    fila["split"] = ""

# ---------------------------------------------- 2. Fotos propias -> test externo
for fila in filas:
    if fila["source"] == "own":
        fila["split"] = "test_propias"

# -------------------------------- 3. Kaggle: agrupar tomas casi iguales y dividir
print("Agrupando fotos de la misma toma (dHash + rafagas)...")
ids = agrupar_similares(filas)
grupos = {}  # id de grupo -> filas
for fila, g in zip(filas, ids):
    grupos.setdefault(g, []).append(fila)
# id legible y estable: nombre (alfabetico) del primer miembro del grupo
for miembros in grupos.values():
    nombre = min(r["filename"] for r in miembros).rsplit(".", 1)[0]
    for r in miembros:
        r["grupo"] = f"g_{nombre}"

estratos = {}  # (fruta, calidad del grupo) -> lista de grupos de Kaggle
for miembros in grupos.values():
    if any(r["source"] == "own" for r in miembros):
        assert all(r["source"] == "own" for r in miembros), \
            f"foto propia parecida a una de Kaggle: {[r['filename'] for r in miembros]}"
        continue  # ya estan en test_propias
    calidades = {r["quality"] for r in miembros}
    calidad = "regular" if "regular" in calidades else Counter(
        r["quality"] for r in miembros).most_common(1)[0][0]
    estratos.setdefault((miembros[0]["fruit"], calidad), []).append(miembros)

azar = random.Random(SEMILLA)
for clave in sorted(estratos):
    calidad = clave[1]

    def peso(miembros):
        # se reparte segun las fotos de la clase del estrato (en un grupo "regular"
        # de 57 fotos puede haber 1 sola regular: lo que importa repartir es esa)
        return sum(r["quality"] == calidad for r in miembros)

    lista = sorted(estratos[clave], key=lambda m: m[0]["grupo"])  # orden fijo
    azar.shuffle(lista)
    # grupos de mayor peso primero (los chicos rellenan al final y ajustan la
    # proporcion); sort es estable, asi que los empates conservan el orden aleatorio
    lista.sort(key=peso, reverse=True)
    n = sum(peso(m) for m in lista)
    meta = {"train": n * FRACCION_TRAIN, "val": n * FRACCION_VAL,
            "test": n * (1 - FRACCION_TRAIN - FRACCION_VAL)}
    actual = {"train": 0, "val": 0, "test": 0}
    for miembros in lista:
        # el grupo va al split al que mas le falta para llegar a su meta
        destino = max(meta, key=lambda s: (meta[s] - actual[s]) / meta[s])
        for r in miembros:
            r["split"] = destino
        actual[destino] += peso(miembros)

# ------------------------ 4. Tamano por terciles de d_eq (cortes con train)
MASCARAS = DATA / "masks.csv"
if MASCARAS.exists():
    with open(MASCARAS, newline="", encoding="utf-8") as f:
        area_mascara = {r["filename"]: float(r["mask_area"]) for r in csv.DictReader(f)}
    faltan = [r["filename"] for r in filas if r["filename"] not in area_mascara]
    if faltan:
        raise SystemExit(f"{len(faltan)} imagenes sin mascara (p. ej. {faltan[0]}). "
                         "Ejecuta antes: python -m src.data.generar_mascaras")
    for r in filas:
        r["mask_area"] = area_mascara[r["filename"]]
        r["d_eq"] = round(2 * np.sqrt(r["mask_area"] / np.pi), 4)
    medida = "d_eq"
else:
    print("[aviso] no existe data/masks.csv: el tamano se calcula con area_ratio")
    medida = "area_ratio"

for fruta in sorted({r["fruit"] for r in filas}):
    valores_train = [float(r[medida]) for r in filas
                     if r["fruit"] == fruta and r["split"] == "train"]
    corte_1, corte_2 = np.percentile(valores_train, [100 / 3, 200 / 3])
    for r in filas:
        if r["fruit"] == fruta:
            d = float(r[medida])
            r["tamano"] = "pequeno" if d <= corte_1 else ("mediano" if d <= corte_2 else "grande")
    print(f"{fruta}: cortes de tamano ({medida}) = {corte_1:.3f} / {corte_2:.3f}")

# ------------------------------------------------------------------ 5. Guardar
columnas = ["filename", "path", "fruit", "quality", "source", "split",
            "grupo", "tamano", "d_eq", "mask_area", "area_ratio", "diam_ratio"]
with open(DATA / "splits.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
    w.writeheader()
    w.writerows(filas)

# ------------------------------------------------------------ 6. Comprobaciones
assert all(r["split"] for r in filas), "hay imagenes sin split"
assert len({r["filename"] for r in filas}) == len(filas), "nombres repetidos"
splits_por_grupo = {}
for r in filas:
    splits_por_grupo.setdefault(r["grupo"], set()).add(r["split"])
assert all(len(s) == 1 for s in splits_por_grupo.values()), "un grupo quedo en dos splits"
print(f"Grupos: {len(splits_por_grupo)} "
      f"({sum(1 for g in grupos.values() if len(g) > 1)} con mas de una foto)")

print(f"\nTotal: {len(filas)} imagenes -> {DATA / 'splits.csv'}\n")
orden = ["train", "val", "test", "test_propias"]
conteo = Counter((r["split"], r["quality"]) for r in filas)
print(f"{'split':14s}{'good':>8s}{'regular':>9s}{'bad':>8s}{'total':>8s}")
for s in orden:
    g, r_, b = (conteo[(s, q)] for q in ("good", "regular", "bad"))
    print(f"{s:14s}{g:8d}{r_:9d}{b:8d}{g + r_ + b:8d}")
print("\nTamano (todas las fotos):", dict(Counter(r["tamano"] for r in filas)))