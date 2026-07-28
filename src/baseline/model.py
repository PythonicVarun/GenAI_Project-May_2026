import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import config


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.MAX_ITER,
                    class_weight=config.CLASS_WEIGHT,
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def softmax_by_question(scores: np.ndarray, n_options: int = 5) -> np.ndarray:
    """Reshape flat pair scores to (n_questions, 5) and softmax each row."""
    logits = scores.reshape(-1, n_options)
    logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return per-question probabilities of shape (n_questions, 5)."""
    scores = np.asarray(pipeline.decision_function(X))
    return softmax_by_question(scores, n_options=len(config.ANSWER_COLS))


@dataclass
class BaselineArtifacts:
    pipeline: Pipeline
    vectorizer: TfidfVectorizer
    val_map3: Optional[float] = None

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaselineArtifacts":
        with open(path, "rb") as f:
            return pickle.load(f)
