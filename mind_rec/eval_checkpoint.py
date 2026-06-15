#!/usr/bin/env python3
"""
Evaluate a saved checkpoint without retraining.

Usage (from mind_rec/ directory):
    python eval_checkpoint.py --checkpoint results/sweep/checkpoints/npa_run001_best.pt \
        --model npa --run_id 1 --best_epoch 4 \
        --lr 1e-4 --dropout 0.2 --num_filters 400 --batch_size 64
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data.dataset import parse_behaviors, parse_news, MINDEvalDataset, MINDTrainDataset
from data.vocab import Vocab
from evaluate import compute_metrics
from models.nrms import NRMS
from models.naml import NAML
from models.lstur import LSTUR
from models.npa import NPA
from torch.utils.data import DataLoader

SEED = 42
VAL_RATIO = 0.1

MODEL_REGISTRY = {"nrms": NRMS, "naml": NAML, "lstur": LSTUR, "npa": NPA}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compute_eval_metrics(model, eval_ds, device):
    model.eval()
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            scores = model(batch).squeeze(0).cpu().numpy()
            labels = batch["labels"].squeeze(0).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)
    return compute_metrics(all_scores, all_labels)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--run_id", type=int, required=True)
    p.add_argument("--best_epoch", type=int, required=True)
    p.add_argument("--best_val_loss", type=float, default=float("nan"))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num_filters", type=int, default=400)
    p.add_argument("--num_heads", type=int, default=20)
    p.add_argument("--head_dim", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lstur_mode", default="ini")
    p.add_argument("--cnn_kernel_size", type=int, default=3)
    p.add_argument("--device", default="mps")
    p.add_argument("--train_dir", default="MINDsmall_train/MINDsmall_train")
    p.add_argument("--dev_dir", default="MINDsmall_dev/MINDsmall_dev")
    p.add_argument("--results_dir", default="results/sweep")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading data...")
    train_news = parse_news(f"{args.train_dir}/news.tsv")
    dev_news   = parse_news(f"{args.dev_dir}/news.tsv")
    all_news   = {**train_news, **dev_news}

    train_behaviors_all = parse_behaviors(f"{args.train_dir}/behaviors.tsv")
    dev_behaviors       = parse_behaviors(f"{args.dev_dir}/behaviors.tsv")

    rng = random.Random(SEED)
    idxs = list(range(len(train_behaviors_all)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * (1.0 - VAL_RATIO))
    train_set = set(idxs[:n_train])
    val_behaviors = [b for i, b in enumerate(train_behaviors_all) if i not in train_set]

    cats       = sorted({v["category"]    for v in all_news.values()})
    subcats    = sorted({v["subcategory"] for v in all_news.values()})
    cat2idx    = {c: i + 1 for i, c in enumerate(cats)}
    subcat2idx = {s: i + 1 for i, s in enumerate(subcats)}

    all_behaviors = train_behaviors_all + dev_behaviors
    users    = sorted({b["user_id"] for b in all_behaviors})
    user2idx = {u: i + 1 for i, u in enumerate(users)}
    num_users = len(user2idx)

    vocab = Vocab()
    vocab.build(
        [n["title"] + " " + n["abstract"] for n in all_news.values()],
        min_freq=1,
    )
    print(f"vocab={len(vocab):,}  users={num_users:,}")

    cfg = Config()
    cfg.model.dropout         = args.dropout
    cfg.model.num_heads       = args.num_heads
    cfg.model.head_dim        = args.head_dim
    cfg.model.num_filters     = args.num_filters
    cfg.model.cnn_kernel_size = args.cnn_kernel_size
    cfg.model.lstur_mode      = args.lstur_mode
    cfg.model.num_users       = num_users
    cfg.train.model_name      = args.model

    base_cfg = Config()
    base_cfg.model.num_users = num_users

    val_eval_ds = MINDEvalDataset(val_behaviors,  all_news, vocab, user2idx, cat2idx, subcat2idx, base_cfg)
    test_ds     = MINDEvalDataset(dev_behaviors,  all_news, vocab, user2idx, cat2idx, subcat2idx, base_cfg)

    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "nrms":
        model = model_cls(len(vocab), cfg)
    elif args.model == "naml":
        model = model_cls(len(vocab), len(cat2idx), len(subcat2idx), cfg)
    elif args.model in ("lstur", "npa"):
        model = model_cls(len(vocab), num_users, cfg)

    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    print("Computing val metrics...")
    val_metrics = compute_eval_metrics(model, val_eval_ds, device)
    print(f"Val  → auc={val_metrics['auc']:.4f}  mrr={val_metrics['mrr']:.4f}  ndcg5={val_metrics['ndcg5']:.4f}  ndcg10={val_metrics['ndcg10']:.4f}")

    print("Computing test metrics...")
    test_metrics = compute_eval_metrics(model, test_ds, device)
    print(f"Test → auc={test_metrics['auc']:.4f}  mrr={test_metrics['mrr']:.4f}  ndcg5={test_metrics['ndcg5']:.4f}  ndcg10={test_metrics['ndcg10']:.4f}")

    hp = {
        "lr": args.lr, "dropout": args.dropout, "batch_size": args.batch_size,
        "num_heads": args.num_heads, "head_dim": args.head_dim,
        "num_filters": args.num_filters, "cnn_kernel_size": args.cnn_kernel_size,
        "lstur_mode": args.lstur_mode,
    }
    ALL_HP_KEYS = ["lr", "dropout", "batch_size", "num_heads", "head_dim", "num_filters", "cnn_kernel_size", "lstur_mode"]
    SUMMARY_FIELDS = (
        ["run_id", "model"] + ALL_HP_KEYS
        + ["best_epoch", "best_val_loss",
           "best_val_auc", "best_val_mrr", "best_val_ndcg5", "best_val_ndcg10",
           "test_auc", "test_mrr", "test_ndcg5", "test_ndcg10",
           "n_params", "checkpoint"]
    )

    summary_row = {"run_id": args.run_id, "model": args.model}
    summary_row.update({k: hp.get(k, "") for k in ALL_HP_KEYS})
    summary_row.update({
        "best_epoch":      args.best_epoch,
        "best_val_loss":   round(args.best_val_loss, 6),
        "best_val_auc":    round(val_metrics["auc"], 6),
        "best_val_mrr":    round(val_metrics["mrr"], 6),
        "best_val_ndcg5":  round(val_metrics["ndcg5"], 6),
        "best_val_ndcg10": round(val_metrics["ndcg10"], 6),
        "test_auc":        round(test_metrics["auc"], 6),
        "test_mrr":        round(test_metrics["mrr"], 6),
        "test_ndcg5":      round(test_metrics["ndcg5"], 6),
        "test_ndcg10":     round(test_metrics["ndcg10"], 6),
        "n_params":        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint":      args.checkpoint,
    })

    summary_path = Path(args.results_dir) / "sweep_summary.csv"
    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Summary appended to {summary_path}")


if __name__ == "__main__":
    main()
