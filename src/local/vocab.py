import logging
import pickle
from typing import Dict, List

from local.utils import clean

logger = logging.getLogger(__name__)


class Vocabulary:
    PAD, UNK, SEP = "<PAD>", "<UNK>", "<SEP>"

    def __init__(self, max_size: int = 20_000):
        self.max_size = max_size
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}

    def build(self, texts: List[str], min_freq: int = 1):
        freq: Dict[str, int] = {}
        for t in texts:
            for w in clean(t).split():
                freq[w] = freq.get(w, 0) + 1

        specials = [self.PAD, self.UNK, self.SEP]
        top = sorted(freq, key=lambda w: freq.get(w, 0), reverse=True)
        top = [w for w in top if freq[w] >= min_freq]
        top = top[: self.max_size - len(specials)]
        for i, w in enumerate(specials + top):
            self.w2i[w] = i
            self.i2w[i] = w

        logger.info("Vocabulary built successfully: size = %d", len(self.w2i))

    @property
    def pad_idx(self):
        return self.w2i[self.PAD]

    @property
    def unk_idx(self):
        return self.w2i[self.UNK]

    @property
    def sep_idx(self):
        return self.w2i[self.SEP]

    def encode(self, text: str, max_len: int) -> List[int]:
        toks = clean(text).split()[:max_len]
        return [self.w2i.get(t, self.unk_idx) for t in toks]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
