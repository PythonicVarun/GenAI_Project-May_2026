import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

os.chdir(REPO_ROOT)

from local.config import (  # noqa: E402
    ANSWER_COLS as CHOICES,
    RETRIEVER_TOP_K,
)

BILSTM = "BiLSTM scorer (from scratch)"
BASELINE = "TF-IDF + Logistic Regression"
GEMMA = "Gemma-4 E4B + LoRA"
ALL_MODELS = [BILSTM, BASELINE, GEMMA]


@dataclass
class Prediction:
    probabilities: np.ndarray  # shape (5,)
    context: str  # retrieved original context
    model_context: str = ""  # what model receives
    note: str = ""

    @property
    def ranking(self) -> list[str]:
        return [CHOICES[i] for i in self.probabilities.argsort()[::-1]]

    @property
    def top3(self) -> str:
        return " ".join(self.ranking[:3])


def build_row(question: str, options: list[str]) -> pd.DataFrame:
    row = {"id": 0, "prompt": question}
    row.update(dict(zip(CHOICES, options)))
    return pd.DataFrame([row])


def unavailable(reason: str) -> Prediction:
    return Prediction(np.full(5, 0.2), "", "", reason)


# Model 1: BiLSTM
@lru_cache()
def _load_bilstm():
    from local.inference import load_artifacts

    return load_artifacts()


@lru_cache()
def _train_rows() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "dataset" / "train.csv")


def retrieved_context(retriever, question: str, top_k: int = 3) -> str:
    from sklearn.metrics.pairwise import cosine_similarity

    from local.utils import clean

    query = retriever.vectorizer.transform([clean(question)])
    order = cosine_similarity(query, retriever.corpus_matrix).flatten().argsort()[::-1]

    train = _train_rows()
    aligned = len(train) == len(retriever.corpus_prompts)

    blocks = []
    for idx in order[:top_k]:
        block = f"Question: {retriever.corpus_prompts[idx]}"
        if aligned:
            row = train.iloc[idx]
            letter = str(row["answer"]).strip().upper()
            block += f"\nAnswer: {letter}) {row[letter]}"
        blocks.append(block)
    return "\n\n".join(blocks)


def run_bilstm(question: str, options: list[str]) -> Prediction:
    import torch
    import torch.nn.functional as F

    from local import config
    from local.dataset import MCQDataset

    model, vocab, retriever = _load_bilstm()
    dataset = MCQDataset(
        build_row(question, options),
        retriever,
        vocab,
        config.MAX_SEQ_LEN,
        is_train=False,
    )
    triples, lengths, _ = dataset[0]

    model.eval()
    with torch.no_grad():
        logits = model(
            triples.unsqueeze(0).to(config.DEVICE),
            lengths.unsqueeze(0).to(config.DEVICE),
        )
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return Prediction(
        probs,
        retrieved_context(retriever, question, top_k=RETRIEVER_TOP_K),
        retriever.retrieve(question, top_k=RETRIEVER_TOP_K),
    )


# Model 2: Gemma-4 + LoRA
def gemma_status():
    return (
        "Gemma-4 needs a CUDA GPU. This deployment is running on CPU, so only the "
        "BiLSTM and the logistic-regression baseline are available."
    )


def run_gemma(question: str, options: list[str]):
    return unavailable(gemma_status())


# Model 3: TF-IDF + logistic regression
@lru_cache()
def _load_baseline():
    from baseline.inference import load_artifacts

    return load_artifacts()


def run_baseline(question: str, options: list[str]) -> Prediction:
    from baseline.inference import probabilities

    artifacts = _load_baseline()
    probs = probabilities(build_row(question, options), artifacts)
    return Prediction(probs[0], "")


RUNNERS = {BILSTM: run_bilstm, BASELINE: run_baseline, GEMMA: run_gemma}


def run(name: str, question: str, options: list[str]) -> Prediction:
    return RUNNERS[name](question, options)
