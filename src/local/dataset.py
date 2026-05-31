from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from local.config import ANSWER_COLS, LABEL2IDX, Q_FRAC
from local.retriever import TFIDFRetriever
from local.utils import extract_core_question
from local.vocab import Vocabulary


def build_triple(
    question: str,
    context: str,
    choice: str,
    vocab: Vocabulary,
    max_len: int,
) -> Tuple[List[int], int]:
    # Encode question and choice first without hard constraints to keep them full
    q_ids = vocab.encode(question, max_len)
    c_ids = vocab.encode(choice, max_len)

    reserved_space = len(q_ids) + len(c_ids) + 2
    if reserved_space < max_len:
        # Give all remaining space to the context
        ctx_limit = max_len - reserved_space
        ctx_ids = vocab.encode(context, ctx_limit)
    else:
        # Fallback: Question + Option exceed max_len.
        # Drop context completely and truncate using Q_FRAC
        ctx_ids = []
        q_limit = max(1, int(max_len * Q_FRAC))
        c_limit = max(1, max_len - q_limit - 2)
        q_ids = vocab.encode(question, q_limit)
        c_ids = vocab.encode(choice, c_limit)

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
            ex_prompt = q if is_train else None
            ex_core = extract_core_question(q) if is_train else None

            ctx = retriever.retrieve(
                q,
                exclude_idx=ex_idx,
                exclude_prompt=ex_prompt,
                exclude_core_question=ex_core,
            )
            label = LABEL2IDX.get(str(row.get("answer", "A")), 0)

            triples, lengths = [], []
            for col in ANSWER_COLS:
                ids, n = build_triple(q, ctx, str(row[col]), vocab, max_len)
                triples.append(ids)
                lengths.append(n)

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
