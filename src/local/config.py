import os
from pathlib import Path

import torch

ANSWER_COLS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {c: i for i, c in enumerate(ANSWER_COLS)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}

Q_FRAC, CTX_FRAC, C_FRAC = 0.40, 0.40, 0.20

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    DATASET_DIR = Path("/kaggle/input/competitions/smart-mcq-solver-challenge")
else:
    DATASET_DIR = Path("dataset")

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found.")

TRAIN_CSV = DATASET_DIR / "train.csv"
if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Training CSV file '{TRAIN_CSV}' not found.")

TEST_CSV = DATASET_DIR / "test.csv"
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Testing CSV file '{TEST_CSV}' not found.")

OUTPUT_DIR = Path("outputs/local")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVER_TOP_K = 3
RETRIEVER_MAX_WORDS = 60

VOCAB_SIZE = 20_000
MAX_SEQ_LEN = 256

EMBED_DIM = 128
HIDDEN_DIM = 192
NUM_LAYERS = 2
DROPOUT = 0.35

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
VAL_FRAC = 0.15
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "varunagnihotri")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "24f2004142-t22026")
