import logging
from typing import Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from baseline import config
from local.utils import clean

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "cos_prompt_option",  # TF-IDF similarity between the question and the option
    "jaccard_prompt_option",  # raw word overlap (a non TF-IDF view of the same idea)
    "option_words",  # length of the option in words
    "length_ratio",  # option length / mean option length of that row
    "is_longest",  # 1 if it is the longest option of the row
    "has_digit",  # 1 if the option contains a number
    "cos_prompt_centered",  # cos_prompt_option minus the row mean
    "cos_prompt_rank",  # within-row rank of cos_prompt_option, scaled to [0, 1]
]


def build_vectorizer(df: pd.DataFrame) -> TfidfVectorizer:
    """Fit a TF-IDF space."""
    corpus = [clean(t) for t in df["prompt"].tolist()]
    for col in config.ANSWER_COLS:
        corpus += [clean(t) for t in df[col].tolist()]

    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
    )
    vectorizer.fit(corpus)
    logger.info(
        "TF-IDF vectorizer fitted on %d documents (vocab size = %d)",
        len(corpus),
        len(vectorizer.vocabulary_),
    )
    return vectorizer


def rowwise_cosine(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    """Cosine similarity"""
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def jaccard(a: str, b: str) -> float:
    """Word-level Jaccard overlap: |A n B| / |A u B|."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_features(
    df: pd.DataFrame, vectorizer: TfidfVectorizer
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Turn a dataframe into a feature matrix and (if available) labels.

    Returns:
    X : shape (n_rows * 5, n_features)
        One row per (question, option) pair, options ordered A..E.
    y : shape (n_rows * 5,)
        1 for the correct option of each question, 0 for the four distractors.
    """
    n = len(df)
    n_opts = len(config.ANSWER_COLS)

    prompts = [clean(str(p)) for p in df["prompt"].tolist()]

    # Options are flattened row-major: [row0_A, row0_B, ..., row0_E, row1_A, ...]
    options = [
        clean(str(row[col])) for _, row in df.iterrows() for col in config.ANSWER_COLS
    ]

    prompt_vecs = cast(csr_matrix, vectorizer.transform(prompts))
    option_vecs = cast(csr_matrix, vectorizer.transform(options))

    # Repeat each question vector five times to align it with its five options
    repeat_idx = np.repeat(np.arange(n), n_opts)
    cos_prompt = rowwise_cosine(cast(csr_matrix, prompt_vecs[repeat_idx]), option_vecs)

    overlap = np.array(
        [
            jaccard(prompts[i], options[i * n_opts + j])
            for i in range(n)
            for j in range(n_opts)
        ]
    )
    n_words = np.array([len(o.split()) for o in options], dtype=float)
    has_digit = np.array([any(ch.isdigit() for ch in o) for o in options], dtype=float)

    # Reshape -> (n_rows, 5)
    cos_prompt_2d = cos_prompt.reshape(n, n_opts)
    n_words_2d = n_words.reshape(n, n_opts)

    mean_words = np.clip(n_words_2d.mean(axis=1, keepdims=True), 1e-6, None)
    length_ratio = n_words_2d / mean_words
    is_longest = (n_words_2d == n_words_2d.max(axis=1, keepdims=True)).astype(float)
    cos_prompt_centered = cos_prompt_2d - cos_prompt_2d.mean(axis=1, keepdims=True)
    cos_prompt_rank = cos_prompt_2d.argsort(axis=1).argsort(axis=1) / (n_opts - 1)

    X = np.column_stack(
        [
            cos_prompt,
            overlap,
            n_words,
            length_ratio.ravel(),
            is_longest.ravel(),
            has_digit,
            cos_prompt_centered.ravel(),
            cos_prompt_rank.ravel(),
        ]
    )

    y = None
    if "answer" in df.columns:
        labels = np.array([config.LABEL2IDX[str(a)] for a in df["answer"]])
        y = np.zeros((n, n_opts))
        y[np.arange(n), labels] = 1.0
        y = y.ravel()

    logger.info("Built feature matrix of shape %s (%d questions)", X.shape, n)
    return X, y
