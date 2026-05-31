import logging
import pickle
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from local.utils import clean, extract_core_question

logger = logging.getLogger(__name__)


def build_row_text(row: pd.Series) -> str:
    """Concatenate prompt + correct choice in 'Question: ... Answer: ...' format."""
    ans_col = str(row["answer"])
    return f"Question: {row['prompt']} Answer: {row[ans_col]}"


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
        self.corpus_prompts: List[str] = []
        self.corpus_core_questions: List[str] = []
        self.corpus_texts: List[str] = []
        self.corpus_matrix = None  # (N, vocab)

    def fit(self, df: pd.DataFrame | pd.Series):
        if isinstance(df, pd.Series):
            df = df.to_frame()

        self.corpus_prompts = df["prompt"].tolist()
        self.corpus_core_questions = df["prompt"].apply(extract_core_question).tolist()
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
        exclude_prompt: Optional[str] = None,
        exclude_core_question: Optional[str] = None,
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

            if (
                exclude_prompt is not None
                and idx < len(self.corpus_prompts)
                and self.corpus_prompts[idx] == exclude_prompt
            ):
                continue

            if (
                exclude_core_question is not None
                and idx < len(self.corpus_core_questions)
                and self.corpus_core_questions[idx] == exclude_core_question
            ):
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
