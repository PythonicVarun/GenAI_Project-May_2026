import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from baseline import config
from baseline.features import FEATURE_NAMES, build_features, build_vectorizer
from baseline.model import BaselineArtifacts, build_classifier, predict_proba
from local.config import IDX2LABEL
from local.git_utils import check_git_status_and_confirm, get_git_commit_url
from local.utils import map_at_3, set_seed

logger = logging.getLogger(__name__)


def compute_metrics(probs: np.ndarray, y: np.ndarray, prefix: str) -> Dict[str, float]:
    true_idx = y.reshape(-1, len(config.ANSWER_COLS)).argmax(axis=1)
    pred_idx = probs.argmax(axis=1)

    ranked = [[IDX2LABEL[i] for i in row.argsort()[::-1][:3]] for row in probs]
    labels = [IDX2LABEL[i] for i in true_idx]

    return {
        f"{prefix}_map3": map_at_3(ranked, labels),
        f"{prefix}_accuracy": float((pred_idx == true_idx).mean()),
        f"{prefix}_macro_f1": float(f1_score(true_idx, pred_idx, average="macro")),
    }


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.SEED)
    order = rng.permutation(len(df))
    n_val = max(1, int(len(df) * config.VAL_FRAC))

    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[order[:n_val]] = True

    df_val = df[val_mask].reset_index(drop=True)
    df_train = df[~val_mask].reset_index(drop=True)
    return df_train, df_val


def log_feature_weights(pipeline: Pipeline) -> None:
    coefs = pipeline.named_steps["clf"].coef_[0]
    for name, weight in zip(FEATURE_NAMES, coefs):
        logger.info("  %-24s %+.4f", name, weight)

    wandb.log({f"coef/{name}": float(w) for name, w in zip(FEATURE_NAMES, coefs)})


def train() -> BaselineArtifacts:
    if not check_git_status_and_confirm():
        raise RuntimeError("Aborted run due to uncommitted files in repository.")

    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logger.info("Initializing Model 3 (TF-IDF features + Logistic Regression)")

    wandb_config = {
        "model": "tfidf_logreg",
        "seed": config.SEED,
        "tfidf_max_features": config.TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": str(config.TFIDF_NGRAM_RANGE),
        "C": config.C,
        "max_iter": config.MAX_ITER,
        "class_weight": config.CLASS_WEIGHT,
        "val_frac": config.VAL_FRAC,
        "n_features": len(FEATURE_NAMES),
    }

    commit_url = get_git_commit_url()
    if commit_url:
        wandb_config["git_commit_url"] = commit_url

    wandb.init(
        entity=config.WANDB_ENTITY,
        project=config.WANDB_PROJECT,
        tags=["model3", "tfidf", "logistic-regression"],
        config=wandb_config,
        notes=f"Git commit: {commit_url}" if commit_url else "No git info available",
    )

    try:
        df = pd.read_csv(config.TRAIN_CSV)
        logger.info(
            "Loaded dataset from %s containing %d samples", config.TRAIN_CSV, len(df)
        )

        df_train, df_val = split_dataframe(df)
        logger.info(
            "Data Split (random): train=%d samples, validation=%d samples",
            len(df_train),
            len(df_val),
        )

        vectorizer = build_vectorizer(df_train)

        logger.info("Building features...")
        X_train, y_train = build_features(df_train, vectorizer)
        X_val, y_val = build_features(df_val, vectorizer)
        assert y_train is not None and y_val is not None, "train.csv needs 'answer'"

        logger.info("Fitting Logistic Regression on %d pairs...", len(X_train))
        pipeline = build_classifier()
        pipeline.fit(X_train, y_train)

        metrics = compute_metrics(predict_proba(pipeline, X_train), y_train, "train")
        metrics.update(compute_metrics(predict_proba(pipeline, X_val), y_val, "val"))

        logger.info(
            "Training finished | Train MAP@3: %.4f | Val MAP@3: %.4f | "
            "Val Accuracy: %.4f | Val Macro-F1: %.4f",
            metrics["train_map3"],
            metrics["val_map3"],
            metrics["val_accuracy"],
            metrics["val_macro_f1"],
        )

        logger.info("Learned feature weights:")
        log_feature_weights(pipeline)

        wandb.log(metrics)
        if wandb.run is not None:
            wandb.run.summary.update(metrics)
            wandb.run.summary["best_val_map3"] = metrics["val_map3"]

        artifacts = BaselineArtifacts(
            pipeline=pipeline,
            vectorizer=vectorizer,
            val_map3=metrics["val_map3"],
        )
        path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
        artifacts.save(path)
        logger.info("Saved artifacts to %s", path)
        return artifacts

    finally:
        wandb.finish()
