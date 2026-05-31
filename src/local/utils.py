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


def extract_core_question(prompt: str) -> str:
    """Extract core question by removing template prefixes and suffixes."""
    prefixes = [
        "Pick the best possible answer:",
        "Determine the correct option:",
        "Select the most accurate option:",
        "Identify the correct statement:",
        "Which of the following is correct?",
        "Choose the correct answer:",
        "Choose the correct option:",
        "Select the correct statement:",
    ]

    suffixes = [
        "among the listed options.",
        "carefully.",
        "based on the given context.",
        "from the following choices.",
    ]

    p = prompt.strip()
    for prefix in prefixes:
        if p.startswith(prefix):
            p = p[len(prefix) :].strip()
            break

    for suffix in suffixes:
        if p.endswith(suffix):
            p = p[: -len(suffix)].strip()
            break
    return p
