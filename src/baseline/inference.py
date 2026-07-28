import logging
import os

import numpy as np
import pandas as pd

from baseline import config
from baseline.features import build_features
from baseline.model import BaselineArtifacts, predict_proba
from local.config import IDX2LABEL

logger = logging.getLogger(__name__)


def probabilities(df: pd.DataFrame, artifacts: BaselineArtifacts) -> np.ndarray:
    X, _ = build_features(df, artifacts.vectorizer)
    return predict_proba(artifacts.pipeline, X)


def predict(df: pd.DataFrame, artifacts: BaselineArtifacts) -> pd.DataFrame:
    probs = probabilities(df, artifacts)
    preds = [" ".join(IDX2LABEL[i] for i in row.argsort()[::-1][:3]) for row in probs]
    return pd.DataFrame({"id": df.get("id", range(len(df))), "Prediction": preds})


def export_probs(
    df: pd.DataFrame, artifacts: BaselineArtifacts, out_path: str
) -> np.ndarray:
    mat = probabilities(df, artifacts)
    np.save(out_path, mat)
    logger.info("Exported probabilities to %s (Shape: %s)", out_path, mat.shape)
    return mat


def load_artifacts() -> BaselineArtifacts:
    path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
    artifacts = BaselineArtifacts.load(path)
    logger.info(
        "Loaded baseline artifacts from %s (Validation MAP@3: %.4f)",
        path,
        artifacts.val_map3 if artifacts.val_map3 is not None else float("nan"),
    )
    return artifacts
