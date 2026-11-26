"""Rutas y constantes globales del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
KAGGLE_RAW_DIR = DATA_DIR / "kaggle_raw"     # dataset Kaggle descomprimido (no versionado)
PROCESSED_DIR = DATA_DIR / "processed"
OWN_DIR = DATA_DIR / "fotos_originales"     # fotos propias originales (respaldo)
MANIFEST_CSV = DATA_DIR / "manifest.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"         # fuente unica de verdad de los splits
MASKS_DIR = DATA_DIR / "masks"               # mascaras GrabCut (src/data/generar_mascaras.py)

EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
IMG_SIZE = (224, 224)  # tamano de data/processed (prepare_dataset.py --size)

FRUITS = ["apple", "banana", "orange"]
QUALITY_CLASSES = ["good", "regular", "bad"]
SIZE_CLASSES = ["pequeno", "mediano", "grande"]  # columna `tamano` de splits.csv
SPLITS = ["train", "val", "test", "test_propias"]
