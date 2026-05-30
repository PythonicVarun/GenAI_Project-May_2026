import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from local import config
from local.config import IDX2LABEL
from local.dataset import MCQDataset, collate_fn
from local.model import BiLSTMScorer
from local.retriever import TFIDFRetriever
from local.vocab import Vocabulary

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict(df, model, retriever, vocab, is_train=False):
    device = config.DEVICE
    model.eval()
    ds = MCQDataset(df, retriever, vocab, config.MAX_SEQ_LEN, is_train=is_train)
    loader = DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    preds = []
    for tri, lng, _ in tqdm(loader, desc="Generating predictions", leave=False):
        probs = F.softmax(model(tri.to(device), lng.to(device)), dim=-1).cpu().numpy()
        for p_row in probs:
            ranked = [IDX2LABEL[i] for i in p_row.argsort()[::-1]]
            preds.append(" ".join(ranked[:3]))

    sub = pd.DataFrame({"id": df.get("id", range(len(df))), "Prediction": preds})
    return sub


@torch.no_grad()
def export_probs(df, model, retriever, vocab, out_path, is_train=False):
    """Export (N, 5) softmax array - input features for Model 3 stacker."""
    device = config.DEVICE
    model.eval()
    ds = MCQDataset(df, retriever, vocab, config.MAX_SEQ_LEN, is_train=is_train)
    loader = DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    probs = []
    for tri, lng, _ in tqdm(loader, desc="Exporting probabilities", leave=False):
        probs.append(
            F.softmax(model(tri.to(device), lng.to(device)), dim=-1).cpu().numpy()
        )

    mat = np.vstack(probs)
    np.save(out_path, mat)
    logger.info("Exported probabilities to %s (Shape: %s)", out_path, mat.shape)
    return mat


def load_artifacts():
    vocab = Vocabulary.load(os.path.join(config.OUTPUT_DIR, "vocab.pkl"))
    retriever = TFIDFRetriever.load(os.path.join(config.OUTPUT_DIR, "retriever.pkl"))
    ckpt = torch.load(
        os.path.join(config.OUTPUT_DIR, "local_model_best.pt"),
        map_location=config.DEVICE,
    )

    model = BiLSTMScorer(
        vocab_size=len(vocab.w2i),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_idx=vocab.pad_idx,
    ).to(config.DEVICE)

    model.load_state_dict(ckpt["model_state"])
    logger.info(
        "Loaded model artifacts successfully from %s (Validation MAP@3: %.4f)",
        config.OUTPUT_DIR,
        ckpt["val_map3"],
    )
    return model, vocab, retriever
