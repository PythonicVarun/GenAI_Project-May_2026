import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from local import config
from local.inference import export_probs, load_artifacts, predict
from local.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("local")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Local Model for Multiple-Choice Question Answering"
    )
    p.add_argument(
        "-m", "--mode", choices=["train", "predict", "export_probs"], default="train"
    )
    p.add_argument("--train_csv", default=config.TRAIN_CSV, type=Path)
    p.add_argument("--test_csv", default=config.TEST_CSV, type=Path)
    p.add_argument("-o", "--output_dir", default=config.OUTPUT_DIR, type=Path)
    args = p.parse_args()

    if args.train_csv is not None:
        config.TRAIN_CSV = args.train_csv
        if not config.TRAIN_CSV.exists():
            raise FileNotFoundError(
                f"Training CSV file '{config.TRAIN_CSV}' not found."
            )

    if args.test_csv is not None:
        config.TEST_CSV = args.test_csv
        if not config.TEST_CSV.exists():
            raise FileNotFoundError(f"Testing CSV file '{config.TEST_CSV}' not found.")

    if args.output_dir is not None:
        config.OUTPUT_DIR = args.output_dir

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if args.mode == "train":
        logger.info("Starting training pipeline...")
        model, vocab, retriever = train()
        df_full = pd.read_csv(config.TRAIN_CSV)
        export_probs(
            df_full,
            model,
            retriever,
            vocab,
            out_path=os.path.join(config.OUTPUT_DIR, "local_model_train_probs.npy"),
            is_train=True,
        )

    elif args.mode == "predict":
        logger.info("Starting inference pipeline...")
        model, vocab, retriever = load_artifacts()
        df_test = pd.read_csv(config.TEST_CSV)
        sub = predict(df_test, model, retriever, vocab, is_train=False)
        out = os.path.join(config.OUTPUT_DIR, "submission_local_model.csv")
        sub.to_csv(out, index=False)
        logger.info("Predictions saved successfully to %s", out)
        print(sub.head())

    elif args.mode == "export_probs":
        logger.info("Starting probabilities export pipeline...")
        model, vocab, retriever = load_artifacts()
        df_test = pd.read_csv(config.TEST_CSV)
        export_probs(
            df_test,
            model,
            retriever,
            vocab,
            out_path=os.path.join(config.OUTPUT_DIR, "local_model_test_probs.npy"),
            is_train=False,
        )
