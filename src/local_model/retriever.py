import logging
import pickle
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from local_model.config import ANSWER_COLS
from local_model.utils import clean

logger = logging.getLogger(__name__)


def build_row_text(row: pd.Series) -> str:
    """Concatenate prompt + all 5 choices into one searchable string."""
    parts = [str(row["prompt"])] + [str(row[c]) for c in ANSWER_COLS]
    return " ".join(parts)


class TFIDFRetriever:
    """
    Indexes training rows (prompt + choices).
    For each query, returns the top-k most similar rows' text as context.
    Pass `exclude_self=True` when querying training rows to prevent leakage.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.corpus_texts: List[str] = []
        self.corpus_matrix = None  # (N, vocab)

    def fit(self, df: pd.DataFrame | pd.Series):
        if isinstance(df, pd.Series):
            df = df.to_frame()

        self.corpus_texts = [
            clean(build_row_text(r))
            for _, r in tqdm(df.iterrows(), total=len(df), desc="Fitting Retriever")
        ]
        self.corpus_matrix = self.vectorizer.fit_transform(self.corpus_texts)
        logger.info(
            "TF-IDF Retriever fitted successfully: indexed %d rows, vocab size = %d",
            len(self.corpus_texts),
            len(self.vectorizer.vocabulary_),
        )

    def retrieve(
        self,
        query: str,
        exclude_idx: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Return retrieved context as a single string."""
        qv = self.vectorizer.transform([clean(query)])
        sim = cosine_similarity(qv, self.corpus_matrix).flatten()

        order = sim.argsort()[::-1]
        picked, ctx_parts = [], []
        for idx in order:
            if len(picked) >= (top_k or self.top_k):
                break

            if exclude_idx is not None and idx == exclude_idx:
                continue

            picked.append(idx)
            ctx_parts.append(self.corpus_texts[idx])

        return " ".join(ctx_parts)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
