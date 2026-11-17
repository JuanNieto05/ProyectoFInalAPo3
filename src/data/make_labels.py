"""Genera el esqueleto de data/annotations/labels.csv a partir de data/own_collection.

Rellena automáticamente fruta, fuente, anotador y fruit_id. La columna `quality`
se etiqueta a mano; `size` y `split` se calculan después (segment.py / split.py).

Uso (desde la raíz del repo):
    python -m src.data.make_labels
"""
import csv

from src.config import ANNOTATIONS_DIR, DATA_DIR, OWN_DIR

OUT = ANNOTATIONS_DIR / "labels.csv"
ANNOTATOR = "Laura"
COLS = ["filename", "fruit", "quality", "size", "source",
        "annotator", "fruit_id", "split", "notes"]


def main() -> None:
    assert not OUT.exists(), "labels.csv ya existe: no lo sobrescribo"
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for p in sorted(OWN_DIR.rglob("*.jpg")):
            w.writerow({
                "filename": p.relative_to(DATA_DIR).as_posix(),
                "fruit": p.parent.name,
                "quality": "",
                "size": "",
                "source": "propia",
                "annotator": ANNOTATOR,
                "fruit_id": p.stem,
                "split": "",
                "notes": "",
            })


if __name__ == "__main__":
    main()