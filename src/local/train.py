import logging
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from local import config
from local.config import ANSWER_COLS, IDX2LABEL
from local.dataset import MCQDataset, collate_fn
from local.model import BiLSTMScorer
from local.retriever import TFIDFRetriever
from local.utils import map_at_3, set_seed
from local.vocab import Vocabulary


def make_scheduler(opt, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - p)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def run_epoch(model, loader, opt, sched, device, clip):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc="Training epoch", leave=False)
    for tri, lng, lbl in pbar:
        tri, lng, lbl = tri.to(device), lng.to(device), lbl.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(tri, lng), lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        total += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for tri, lng, lbl in pbar:
        tri, lng, lbl = tri.to(device), lng.to(device), lbl.to(device)
        logits = model(tri, lng)
        loss = F.cross_entropy(logits, lbl)
        total_loss += loss.item()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        for p_row, l in zip(probs, lbl.cpu().tolist()):  # noqa: E741 - l is label index
            ranked = [IDX2LABEL[i] for i in p_row.argsort()[::-1]]
            all_preds.append(ranked[:3])
            all_labels.append(IDX2LABEL[l])

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader), map_at_3(all_preds, all_labels)


logger = logging.getLogger(__name__)


def train():
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    logger.info("Initializing Model 1 (TF-IDF + BiLSTM Scorer) on device: %s", device)

    wandb_config = {
        "seed": config.SEED,
        "vocab_size": config.VOCAB_SIZE,
        "max_seq_len": config.MAX_SEQ_LEN,
        "embed_dim": config.EMBED_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "lr": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "grad_clip": config.GRAD_CLIP,
        "warmup_ratio": config.WARMUP_RATIO,
        "val_frac": config.VAL_FRAC,
        "device": str(config.DEVICE),
        "retriever_top_k": config.RETRIEVER_TOP_K,
        "retriever_max_words": config.RETRIEVER_MAX_WORDS,
    }

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "24f2004142-t22026"),
        config=wandb_config,
    )

    try:
        df = pd.read_csv(config.TRAIN_CSV)
        logger.info(
            "Loaded dataset from %s containing %d samples", config.TRAIN_CSV, len(df)
        )

        logger.info("Building vocabulary from prompts & answer choices...")
        all_texts = df["prompt"].tolist()
        for c in ANSWER_COLS:
            all_texts += df[c].tolist()
        vocab = Vocabulary(config.VOCAB_SIZE)
        vocab.build(all_texts, min_freq=1)
        vocab.save(os.path.join(config.OUTPUT_DIR, "vocab.pkl"))

        # Split the dataset before fitting the training retriever to avoid data leakage
        df_shuffled = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        n_val = max(1, int(len(df_shuffled) * config.VAL_FRAC))
        df_val = df_shuffled.iloc[:n_val].reset_index(drop=True)
        df_train = df_shuffled.iloc[n_val:].reset_index(drop=True)
        logger.info(
            "Data Split: train=%d samples, validation=%d samples",
            len(df_train),
            len(df_val),
        )

        logger.info("Fitting TF-IDF retriever on train split...")
        train_retriever = TFIDFRetriever(top_k=config.RETRIEVER_TOP_K)
        train_retriever.fit(df_train)

        logger.info("Assembling train and validation datasets...")
        train_ds = MCQDataset(
            df_train, train_retriever, vocab, config.MAX_SEQ_LEN, is_train=True
        )
        val_ds = MCQDataset(
            df_val, train_retriever, vocab, config.MAX_SEQ_LEN, is_train=False
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        logger.info("Initializing model...")
        model = BiLSTMScorer(
            vocab_size=len(vocab.w2i),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_idx=vocab.pad_idx,
        ).to(device)
        logger.info(
            "Model initialized with %s trainable parameters",
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        )

        opt = torch.optim.AdamW(
            model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
        )
        total = config.EPOCHS * len(train_loader)
        sched = make_scheduler(opt, int(total * config.WARMUP_RATIO), total)

        best_map3 = 0.0
        ckpt_path = os.path.join(config.OUTPUT_DIR, "local_model_best.pt")

        logger.info("Starting training loop for %d epochs...", config.EPOCHS)

        for ep in range(1, config.EPOCHS + 1):
            tr_loss = run_epoch(
                model, train_loader, opt, sched, device, config.GRAD_CLIP
            )
            va_loss, va_map3 = evaluate(model, val_loader, device)

            is_best = va_map3 > best_map3
            star = " (Best :)" if is_best else ""
            logger.info(
                "Epoch %d/%d completed | Train Loss: %.4f | Val Loss: %.4f | Val MAP@3: %.4f%s",  # noqa: E501
                ep,
                config.EPOCHS,
                tr_loss,
                va_loss,
                va_map3,
                star,
            )

            # Log metrics to WandB
            wandb.log(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_map3": va_map3,
                    "lr": opt.param_groups[0]["lr"],
                }
            )

            if is_best:
                best_map3 = va_map3
                torch.save(
                    {
                        "epoch": ep,
                        "model_state": model.state_dict(),
                        "val_map3": va_map3,
                    },
                    ckpt_path,
                )
                if wandb.run is not None:
                    wandb.run.summary["best_val_map3"] = best_map3
                else:
                    logger.warning(
                        "WandB run not initialized. Best Val MAP@3 will not be logged"
                        " to WandB summary. BTW this is weird!!"
                    )

        logger.info(
            "Training finished. Best Val MAP@3: %.4f. Saved checkpoint to %s",
            best_map3,
            ckpt_path,
        )

        # Fit the final retriever on the full dataset for saving (used in test time)
        logger.info("Fitting final TF-IDF retriever on full dataset...")
        retriever = TFIDFRetriever(top_k=config.RETRIEVER_TOP_K)
        retriever.fit(df)
        retriever.save(os.path.join(config.OUTPUT_DIR, "retriever.pkl"))
        return model, vocab, retriever

    finally:
        wandb.finish()
