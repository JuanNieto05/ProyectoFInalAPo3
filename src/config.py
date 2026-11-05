"""Rutas y constantes globales del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OWN_DIR = DATA_DIR / "own_collection"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
IMG_SIZE = (128, 128)

# Por definir tras el EDA (fase 2)
QUALITY_CLASSES: list[str] = []
SIZE_CLASSES = ["pequeño", "mediano", "grande"]
