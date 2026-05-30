import re
from typing import List

import numpy as np
import torch


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def clean(text: str) -> str:
    text = str(text).lower()  # type: ignore
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def map_at_3(preds: List[List[str]], labels: List[str]) -> float:
    score = 0.0
    for pred, label in zip(preds, labels):
        for k, p in enumerate(pred[:3], 1):
            if p == label:
                score += 1.0 / k
                break

    return score / max(1, len(labels))
