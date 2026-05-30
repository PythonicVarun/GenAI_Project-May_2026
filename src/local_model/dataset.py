from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from local_model.config import ANSWER_COLS, CTX_FRAC, LABEL2IDX, Q_FRAC
from local_model.retriever import TFIDFRetriever
from local_model.vocab import Vocabulary


def build_triple(
    question: str,
    context: str,
    choice: str,
    vocab: Vocabulary,
    max_len: int,
) -> Tuple[List[int], int]:
    q_max = max(1, int(max_len * Q_FRAC))
    ctx_max = max(1, int(max_len * CTX_FRAC))
    c_max = max(1, max_len - q_max - ctx_max - 2)

    q_ids = vocab.encode(question, q_max)
    ctx_ids = vocab.encode(context, ctx_max)
    c_ids = vocab.encode(choice, c_max)

    ids = q_ids + [vocab.sep_idx] + ctx_ids + [vocab.sep_idx] + c_ids
    ids = ids[:max_len]
    length = len(ids)
    ids += [vocab.pad_idx] * (max_len - len(ids))
    return ids, length


class MCQDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame | pd.Series,
        retriever: TFIDFRetriever,
        vocab: Vocabulary,
        max_len: int,
        is_train: bool = True,
        train_index: Optional[pd.DataFrame] = None,
    ):
        self.records = []

        desc = "Building train dataset" if is_train else "Building validation dataset"
        for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
            q = str(row["prompt"])
            ex_idx = pos if is_train else None
            ctx = retriever.retrieve(q, exclude_idx=ex_idx)
            label = LABEL2IDX.get(str(row.get("answer", "A")), 0)

            triples, lengths = [], []
            for col in ANSWER_COLS:
                ids, l = build_triple(q, ctx, str(row[col]), vocab, max_len)
                triples.append(ids)
                lengths.append(l)

            self.records.append(
                {
                    "triples": triples,
                    "lengths": lengths,
                    "label": label,
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return (
            torch.tensor(r["triples"], dtype=torch.long),
            torch.tensor(r["lengths"], dtype=torch.long),
            torch.tensor(r["label"], dtype=torch.long),
        )


def collate_fn(batch):
    triples, lengths, labels = zip(*batch)
    return torch.stack(triples), torch.stack(lengths), torch.stack(labels)
