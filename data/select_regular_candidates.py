"""
select_regular_candidates.py
----------------------------
Propone imagenes de Kaggle que podrian ser "regular" (la clase intermedia, que
Kaggle no trae) y deja que ustedes las confirmen visualmente.

Idea:
  1. Entrena, por fruta, un clasificador simple good vs bad sobre Kaggle usando
     rasgos de color de la fruta (histogramas HSV, fraccion de pixeles oscuros).
     Las probabilidades son "out-of-fold" (validacion cruzada), asi que cada
     imagen se puntua con un modelo que NO la vio.
  2. Las imagenes donde el modelo duda (p_bad cercana a 0.5) son candidatas a
     "regular". Se combina (70/30) con la cercania al promedio de las fotos
     propias "regular" de esa fruta, si existen.
  3. Se copian las N mejores candidatas por fruta a data/review/<fruta>/.
  4. se revisa esa carpeta y se borran las que NO son regular.
  5. Con --apply, las que quedaron se mueven a data/processed/regular/ y se
     actualiza manifest.csv (columna relabeled_from).

Uso (desde la raiz del proyecto):
    python data/select_regular_candidates.py                 # paso 1: genera candidatas
    # ... revisar y borrar en data/review/ las que no son regular ...
    python data/select_regular_candidates.py --apply         # paso 2: aplica

Dependencias: numpy, pillow, scikit-learn.
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEP = "__"


def features(path: Path) -> np.ndarray:
    """Rasgos de color de los pixeles de la fruta (fondo uniforme descartado)."""
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    mask = np.linalg.norm(rgb - bg, axis=2) > 40
    if mask.sum() < 50:  # segmentacion fallida: usa toda la imagen
        mask = np.ones(mask.shape, dtype=bool)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)[mask]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    hh = np.histogram(h, bins=12, range=(0, 256))[0]
    hs = np.histogram(s, bins=8, range=(0, 256))[0]
    hv = np.histogram(v, bins=8, range=(0, 256))[0]
    n = len(h)
    hist = np.concatenate([hh, hs, hv]) / n
    stats = [s.mean() / 255, s.std() / 255, v.mean() / 255, v.std() / 255,
             (v < 60).mean(),          # fraccion muy oscura (manchas, podredumbre)
             ((v < 110) & (s > 60)).mean()]  # fraccion marron/oscura saturada
    return np.concatenate([hist, stats]).astype(np.float32)


def read_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return list(r), r.fieldnames


def write_manifest(path: Path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def select(data: Path, n_per_fruit: int, force: bool):
    proc, review = data / "processed", data / "review"
    rows, _ = read_manifest(data / "manifest.csv")
    if review.exists():
        if not force:
            raise SystemExit(f"{review} ya existe. Revisa/aplica primero o usa --force para regenerar.")
        shutil.rmtree(review)

    for fruit in sorted({r["fruit"] for r in rows}):
        kag = [r for r in rows if r["fruit"] == fruit and r["source"] == "kaggle"
               and r["quality"] in ("good", "bad")]
        ys = np.array([1 if r["quality"] == "bad" else 0 for r in kag])
        if len(kag) < 20 or len(set(ys)) < 2:
            print(f"[aviso] {fruit}: no hay suficientes imagenes de Kaggle good/bad, la omito")
            continue
        X = np.stack([features(proc / r["path"]) for r in kag])

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        p_bad = cross_val_predict(model, X, ys, cv=5, method="predict_proba")[:, 1]
        acc = ((p_bad > 0.5) == ys).mean()
        print(f"{fruit}: {len(kag)} imagenes Kaggle, accuracy good/bad (CV) = {acc:.3f}")

        score = np.abs(p_bad - 0.5)                      # menor = mas dudosa
        rank_u = score.argsort().argsort() / len(kag)
        own_reg = [r for r in rows if r["fruit"] == fruit and r["source"] == "own" and r["quality"] == "regular"]
        if own_reg:
            Xo = np.stack([features(proc / r["path"]) for r in own_reg])
            sc = StandardScaler().fit(X)
            d = np.linalg.norm(sc.transform(X) - sc.transform(Xo).mean(axis=0), axis=1)
            rank_d = d.argsort().argsort() / len(kag)
            combined = 0.7 * rank_u + 0.3 * rank_d
        else:
            print(f"  [aviso] {fruit}: sin fotos propias 'regular'; solo se usa la incertidumbre")
            combined = rank_u

        out = review / fruit
        out.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(np.argsort(combined)[:n_per_fruit], start=1):
            r = kag[i]
            shutil.copy2(proc / r["path"], out / f"{rank:03d}{SEP}{r['filename']}")
        print(f"  -> {min(n_per_fruit, len(kag))} candidatas en {out}")

    print(f"\nAhora revisen {review}: BORREN las imagenes que NO sean 'regular'.\n"
          f"Luego ejecuten de nuevo con --apply.")


def apply(data: Path):
    proc, review = data / "processed", data / "review"
    if not review.exists():
        raise SystemExit(f"No existe {review}. Ejecuta primero sin --apply.")
    rows, fields = read_manifest(data / "manifest.csv")
    by_name = {r["filename"]: r for r in rows}
    if "relabeled_from" not in fields:
        fields = fields + ["relabeled_from"]
        for r in rows:
            r["relabeled_from"] = ""
    (proc / "regular").mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in sorted(review.rglob("*.jpg")):
        orig = f.name.split(SEP, 1)[1]
        r = by_name.get(orig)
        if r is None or r["quality"] == "regular":
            continue
        old_q = r["quality"]
        new_name = orig.replace(f"_{old_q}_kaggle_", f"_regular_kaggle_{old_q}_")
        src = proc / r["path"]
        if not src.exists():
            continue
        shutil.move(str(src), str(proc / "regular" / new_name))
        r.update(filename=new_name, path=f"regular/{new_name}", quality="regular", relabeled_from=old_q)
        moved += 1

    write_manifest(data / "manifest.csv", rows, fields)
    print(f"Reetiquetadas como 'regular': {moved} imagenes (movidas a {proc / 'regular'}).")
    counts = {}
    for r in rows:
        k = (r["source"], r["quality"])
        counts[k] = counts.get(k, 0) + 1
    for k in sorted(counts):
        print(f"  {k[0]:7s} {k[1]:8s} {counts[k]}")
    print("Puedes borrar data/review si ya no la necesitas.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="carpeta con processed/ y manifest.csv")
    ap.add_argument("--n", type=int, default=150, help="candidatas por fruta")
    ap.add_argument("--apply", action="store_true", help="aplica la revision hecha en data/review")
    ap.add_argument("--force", action="store_true", help="regenera data/review si ya existe")
    a = ap.parse_args()
    (apply if a.apply else lambda d: select(d, a.n, a.force))(Path(a.data))


if __name__ == "__main__":
    main()