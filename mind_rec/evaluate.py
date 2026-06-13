from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score


def _dcg(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = labels[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float((gains / discounts).sum())


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    ideal = _dcg(labels, labels, k)
    return _dcg(scores, labels, k) / ideal if ideal > 0 else 0.0


def mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    for rank, idx in enumerate(np.argsort(scores)[::-1], 1):
        if labels[idx] == 1:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    all_scores: List[np.ndarray], all_labels: List[np.ndarray]
) -> dict:
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for scores, labels in zip(all_scores, all_labels):
        pos = labels.sum()
        if pos == 0 or pos == len(labels):
            continue
        aucs.append(roc_auc_score(labels, scores))
        mrrs.append(mrr(scores, labels))
        ndcg5s.append(ndcg_at_k(scores, labels, 5))
        ndcg10s.append(ndcg_at_k(scores, labels, 10))
    return {
        "auc": float(np.mean(aucs)),
        "mrr": float(np.mean(mrrs)),
        "ndcg5": float(np.mean(ndcg5s)),
        "ndcg10": float(np.mean(ndcg10s)),
        "n_impressions": len(aucs),
    }
