import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .vocab import Vocab


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_news(news_path: str) -> Dict[str, dict]:
    """Columns: news_id  category  subcategory  title  abstract  url  entities..."""
    news: Dict[str, dict] = {}
    with open(news_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            news[parts[0]] = {
                "category": parts[1],
                "subcategory": parts[2],
                "title": parts[3],
                "abstract": parts[4] if len(parts) > 4 else "",
            }
    return news


def parse_behaviors(path: str) -> List[dict]:
    """Columns: index  user_id  time  history  impressions"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            history = parts[3].split() if parts[3].strip() else []
            impressions = parts[4].split()
            pos = [i.split("-")[0] for i in impressions if i.endswith("-1")]
            neg = [i.split("-")[0] for i in impressions if i.endswith("-0")]
            rows.append({
                "user_id": parts[1],
                "history": history,
                "pos": pos,
                "neg": neg,
                "impressions": impressions,
            })
    return rows


# ---------------------------------------------------------------------------
# Helper – shared news encoding
# ---------------------------------------------------------------------------

def _encode_news(nid: str, news: Dict[str, dict], vocab: Vocab, cat2idx, subcat2idx, cfg) -> dict:
    n = news.get(nid, {"category": "", "subcategory": "", "title": "", "abstract": ""})
    return {
        "title": vocab.encode(n["title"], cfg.data.max_title_len),
        "abstract": vocab.encode(n["abstract"], cfg.data.max_abstract_len),
        "category": cat2idx.get(n["category"], 0),
        "subcategory": subcat2idx.get(n["subcategory"], 0),
    }


def _encode_history(history: List[str], news, vocab, cat2idx, subcat2idx, cfg) -> dict:
    H = cfg.data.max_history
    hist = history[-H:]
    pad_len = H - len(hist)
    mask = [True] * len(hist) + [False] * pad_len
    padded = hist + [""] * pad_len
    enc = [_encode_news(nid, news, vocab, cat2idx, subcat2idx, cfg) for nid in padded]
    return {
        "titles": [e["title"] for e in enc],
        "abstracts": [e["abstract"] for e in enc],
        "categories": [e["category"] for e in enc],
        "subcategories": [e["subcategory"] for e in enc],
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Train dataset  – one (pos, neg_list) sample per positive impression
# ---------------------------------------------------------------------------

class MINDTrainDataset(Dataset):
    def __init__(self, behaviors, news, vocab: Vocab, user2idx, cat2idx, subcat2idx, cfg):
        self.news = news
        self.vocab = vocab
        self.user2idx = user2idx
        self.cat2idx = cat2idx
        self.subcat2idx = subcat2idx
        self.cfg = cfg

        self.samples: List[tuple] = []
        for b in behaviors:
            uid = b["user_id"]
            hist = b["history"]
            negs_pool = b["neg"]
            for pos_nid in b["pos"]:
                negs = _sample_negatives(negs_pool, cfg.data.neg_samples)
                self.samples.append((uid, hist, pos_nid, negs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, hist, pos_nid, neg_nids = self.samples[idx]
        candidates = [pos_nid] + neg_nids

        history = _encode_history(hist, self.news, self.vocab, self.cat2idx, self.subcat2idx, self.cfg)
        cands = [_encode_news(nid, self.news, self.vocab, self.cat2idx, self.subcat2idx, self.cfg) for nid in candidates]

        return {
            "user_idx": torch.tensor(self.user2idx.get(uid, 0), dtype=torch.long),
            "history_titles": torch.tensor(history["titles"], dtype=torch.long),
            "history_abstracts": torch.tensor(history["abstracts"], dtype=torch.long),
            "history_categories": torch.tensor(history["categories"], dtype=torch.long),
            "history_subcategories": torch.tensor(history["subcategories"], dtype=torch.long),
            "history_mask": torch.tensor(history["mask"], dtype=torch.bool),
            "cand_titles": torch.tensor([c["title"] for c in cands], dtype=torch.long),
            "cand_abstracts": torch.tensor([c["abstract"] for c in cands], dtype=torch.long),
            "cand_categories": torch.tensor([c["category"] for c in cands], dtype=torch.long),
            "cand_subcategories": torch.tensor([c["subcategory"] for c in cands], dtype=torch.long),
            "label": torch.tensor(0, dtype=torch.long),   # positive is always index 0
        }


# ---------------------------------------------------------------------------
# Eval dataset  – one full impression group, variable candidate count
# ---------------------------------------------------------------------------

class MINDEvalDataset(Dataset):
    def __init__(self, behaviors, news, vocab: Vocab, user2idx, cat2idx, subcat2idx, cfg):
        self.news = news
        self.vocab = vocab
        self.user2idx = user2idx
        self.cat2idx = cat2idx
        self.subcat2idx = subcat2idx
        self.cfg = cfg
        self.samples = [b for b in behaviors if b["pos"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b = self.samples[idx]
        news_ids = [imp.split("-")[0] for imp in b["impressions"]]
        labels = [int(imp.split("-")[1]) for imp in b["impressions"]]

        history = _encode_history(b["history"], self.news, self.vocab, self.cat2idx, self.subcat2idx, self.cfg)
        cands = [_encode_news(nid, self.news, self.vocab, self.cat2idx, self.subcat2idx, self.cfg) for nid in news_ids]

        return {
            "user_idx": torch.tensor(self.user2idx.get(b["user_id"], 0), dtype=torch.long),
            "history_titles": torch.tensor(history["titles"], dtype=torch.long),
            "history_abstracts": torch.tensor(history["abstracts"], dtype=torch.long),
            "history_categories": torch.tensor(history["categories"], dtype=torch.long),
            "history_subcategories": torch.tensor(history["subcategories"], dtype=torch.long),
            "history_mask": torch.tensor(history["mask"], dtype=torch.bool),
            "cand_titles": torch.tensor([c["title"] for c in cands], dtype=torch.long),
            "cand_abstracts": torch.tensor([c["abstract"] for c in cands], dtype=torch.long),
            "cand_categories": torch.tensor([c["category"] for c in cands], dtype=torch.long),
            "cand_subcategories": torch.tensor([c["subcategory"] for c in cands], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sample_negatives(neg_pool: List[str], k: int) -> List[str]:
    if not neg_pool:
        return [""] * k
    if len(neg_pool) >= k:
        return random.sample(neg_pool, k)
    # repeat pool to reach k samples
    return (neg_pool * (k // len(neg_pool) + 1))[:k]
