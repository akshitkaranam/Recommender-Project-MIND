import re
from collections import Counter
from typing import Dict, List

import numpy as np
import torch


def tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


class Vocab:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build(self, texts: List[str], min_freq: int = 1) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = tokenize(text)[:max_len]
        ids = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.word2idx)

    def load_glove(self, glove_path: str, emb_dim: int) -> torch.Tensor:
        rng = np.random.default_rng(42)
        emb = np.zeros((len(self), emb_dim), dtype=np.float32)
        emb[2:] = rng.normal(0, 0.1, (len(self) - 2, emb_dim))
        found = 0
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                idx = self.word2idx.get(parts[0])
                if idx is not None:
                    emb[idx] = np.array(parts[1:], dtype=np.float32)
                    found += 1
        print(f"GloVe: {found}/{len(self)} vocab words found")
        return torch.from_numpy(emb)
