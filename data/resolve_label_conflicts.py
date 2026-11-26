"""
resolve_label_conflicts.py
--------------------------
Resuelve las parejas de fotos casi identicas que quedaron con ETIQUETAS DISTINTAS
despues de find_duplicates.py (ese script solo elimina duplicados con la misma
etiqueta y deja los conflictos "para decidir a mano").

Por que importa:
  - Dos imagenes casi iguales con clases distintas son ruido de etiqueta.
  - Si caen en splits distintos (train / test) hay fuga de informacion.

Regla aplicada (todas las parejas en conflicto son "regular" vs "good"/"bad"):
  se CONSERVA la imagen "regular" (fue revisada a mano al reetiquetar, ver
  select_regular_candidates.py) y se ELIMINA su gemela good/bad. Asi no se
  pierde ninguna imagen de la clase minoritaria.

Las imagenes eliminadas se mueven a data/conflicts_removed/ (no versionada) y
quedan registradas en data/conflicts_removed.csv (versionado, evidencia).

Uso (desde la raiz del proyecto), y luego volver a generar los splits:
    python data/resolve_label_conflicts.py
    python data/make_splits.py
"""
import csv
import shutil
from pathlib import Path

DATA = Path("data")
PROCESSED = DATA / "processed"
DESTINO = DATA / "conflicts_removed"

# ---------------------------------------------------------------- 1. Leer datos
with open(DATA / "manifest.csv", newline="", encoding="utf-8") as f:
    lector = csv.DictReader(f)
    campos = lector.fieldnames
    filas = list(lector)
por_nombre = {r["filename"]: r for r in filas}

with open(DATA / "duplicates.csv", newline="", encoding="utf-8") as f:
    pares = list(csv.DictReader(f))

# ------------------------------------------- 2. Elegir que imagen sale de cada par
eliminar = {}  # filename -> (gemela conservada, distancia)
for p in pares:
    a, b = por_nombre.get(p["foto_a"]), por_nombre.get(p["foto_b"])
    if a is None or b is None:            # alguna ya no esta en el dataset
        continue
    if a["quality"] == b["quality"] and a["fruit"] == b["fruit"]:
        continue                          # no es conflicto
    if a["quality"] == "regular" and b["quality"] != "regular":
        sale, queda = b, a
    elif b["quality"] == "regular" and a["quality"] != "regular":
        sale, queda = a, b
    else:
        raise SystemExit(f"Conflicto sin 'regular', decidir a mano: {p}")
    eliminar.setdefault(sale["filename"], (queda["filename"], p["distancia"]))

# ---------------------------------------------------------- 3. Mover y registrar
DESTINO.mkdir(exist_ok=True)
for nombre in eliminar:
    origen = PROCESSED / por_nombre[nombre]["path"]
    if origen.exists():
        shutil.move(str(origen), str(DESTINO / nombre))

registro = DATA / "conflicts_removed.csv"
with open(registro, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["eliminada", "calidad_eliminada", "conservada", "distancia", "fuente"])
    for nombre, (queda, dist) in sorted(eliminar.items()):
        r = por_nombre[nombre]
        w.writerow([nombre, r["quality"], queda, dist, r["source"]])

restantes = [r for r in filas if r["filename"] not in eliminar]
with open(DATA / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(restantes)

print(f"Eliminadas por conflicto de etiqueta: {len(eliminar)} -> {registro}")
print(f"Imagenes restantes en manifest: {len(restantes)}")
