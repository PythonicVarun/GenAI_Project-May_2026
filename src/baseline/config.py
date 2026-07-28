from pathlib import Path

from local.config import (
    ANSWER_COLS,
    IDX2LABEL,
    LABEL2IDX,
    SEED,
    TEST_CSV,
    TRAIN_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

OUTPUT_DIR = Path("outputs/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)

C = 1.0
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"

VAL_FRAC = 0.15


__all__ = [
    "ANSWER_COLS",
    "IDX2LABEL",
    "LABEL2IDX",
    "SEED",
    "TEST_CSV",
    "TRAIN_CSV",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "OUTPUT_DIR",
    "TFIDF_MAX_FEATURES",
    "TFIDF_NGRAM_RANGE",
    "C",
    "MAX_ITER",
    "CLASS_WEIGHT",
    "VAL_FRAC",
]
