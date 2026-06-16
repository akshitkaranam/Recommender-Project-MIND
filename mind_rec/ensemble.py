#!/usr/bin/env python3
"""
Ensemble inference from sweep results.

Loads the best checkpoint from each run, weights by a chosen strategy,
and evaluates ensemble on the held-out test set (MINDsmall_dev).

Usage:
    python ensemble.py                        # weight by test AUC (quick, slight leakage)
    python ensemble.py --weight_by val_opt    # optimise weights on val set (recommended)
    python ensemble.py --weight_by uniform    # equal weights
    python ensemble.py --weight_by val_auc    # proportional to best val AUC (no leakage)
    python ensemble.py --models nrms naml     # only these model families
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import parse_behaviors, parse_news, MINDEvalDataset
from data.vocab import Vocab
from evaluate import compute_metrics
from models.nrms import NRMS
from models.naml import NAML
from models.lstur import LSTUR
from models.npa import NPA

SEED = 42
VAL_RATIO = 0.1

MODEL_REGISTRY = {
    "nrms": NRMS,
    "naml": NAML,
    "lstur": LSTUR,
    "npa": NPA,
}


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def load_checkpoint(ckpt_path: str, model: nn.Module, device: torch.device) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


def build_model(model_name: str, vocab_size: int, num_cats: int,
                num_subcats: int, num_users: int, cfg: Config,
                hp: dict = None) -> nn.Module:
    if hp:
        if hp.get("lr"):
            cfg.train.lr = float(hp["lr"])
        if hp.get("dropout"):
            cfg.model.dropout = float(hp["dropout"])
        if hp.get("num_heads"):
            cfg.model.num_heads = int(float(hp["num_heads"]))
        if hp.get("head_dim"):
            cfg.model.head_dim = int(float(hp["head_dim"]))
        if hp.get("num_filters"):
            cfg.model.num_filters = int(float(hp["num_filters"]))
        if hp.get("cnn_kernel_size"):
            cfg.model.cnn_kernel_size = int(float(hp["cnn_kernel_size"]))
        if hp.get("lstur_mode"):
            cfg.model.lstur_mode = hp["lstur_mode"]
        if hp.get("batch_size"):
            cfg.train.batch_size = int(float(hp["batch_size"]))

    cfg.model.num_users = num_users
    if model_name == "nrms":
        return NRMS(vocab_size, cfg)
    if model_name == "naml":
        return NAML(vocab_size, num_cats, num_subcats, cfg)
    if model_name == "lstur":
        return LSTUR(vocab_size, num_users, cfg)
    if model_name == "npa":
        return NPA(vocab_size, num_users, cfg)
    raise ValueError(f"Unknown model: {model_name}")


def parse_sweep_summary(csv_path: str) -> List[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sweep summary not found: {csv_path}")
    runs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    if not runs:
        raise ValueError("No runs found in sweep summary")
    print(f"  Loaded {len(runs)} runs from sweep summary")
    return runs


class ModelEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], weights: np.ndarray):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        assert len(models) == len(weights)

    def forward(self, batch: dict) -> torch.Tensor:
        stacked = torch.stack([m(batch) for m in self.models], dim=0)  # (M, B, K)
        return (stacked * self.weights.view(-1, 1, 1)).sum(dim=0)


def collect_scores(
    models: List[nn.Module],
    dataset,
    device: torch.device,
    desc: str = "collecting scores",
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    """
    Run every model over the dataset in a single pass.
    Returns (per_model_scores, labels_list) where
      per_model_scores[m][i] = 1-D score array for impression i, model m
      labels_list[i]         = 1-D label array for impression i
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    n_models = len(models)
    per_model_scores: List[List[np.ndarray]] = [[] for _ in range(n_models)]
    labels_list: List[np.ndarray] = []

    for m in models:
        m.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {desc}", leave=False):
            batch = _to_device(batch, device)
            labels_list.append(batch["labels"].squeeze(0).cpu().numpy())
            for m_idx, model in enumerate(models):
                per_model_scores[m_idx].append(
                    model(batch).squeeze(0).cpu().numpy()
                )

    return per_model_scores, labels_list


def optimize_weights(
    per_model_scores: List[List[np.ndarray]],
    labels_list: List[np.ndarray],
    n_models: int,
) -> Tuple[np.ndarray, float]:
    """
    Find weights (summing to 1, all >= 0) that maximise AUC on the given
    score/label set.  Uses softmax parameterisation so the search space is
    unconstrained, then Nelder-Mead.
    """
    def neg_auc(theta: np.ndarray) -> float:
        w = scipy_softmax(theta)
        ens = [
            sum(w[m] * per_model_scores[m][i] for m in range(n_models))
            for i in range(len(labels_list))
        ]
        return -compute_metrics(ens, labels_list)["auc"]

    print(f"  Optimising weights over {n_models} models on val set "
          f"({len(labels_list):,} impressions)...")
    theta0 = np.zeros(n_models, dtype=np.float64)
    result = minimize(
        neg_auc, theta0, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-5, "disp": False},
    )
    weights = scipy_softmax(result.x).astype(np.float32)
    val_auc = -result.fun
    return weights, val_auc


def evaluate(ensemble: ModelEnsemble, dataset, device: torch.device) -> dict:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []
    ensemble.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            batch = _to_device(batch, device)
            all_scores.append(ensemble(batch).squeeze(0).cpu().numpy())
            all_labels.append(batch["labels"].squeeze(0).cpu().numpy())
    return compute_metrics(all_scores, all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                        choices=list(MODEL_REGISTRY))
    parser.add_argument("--weight_by",
                        choices=["uniform", "auc", "mrr", "ndcg5", "val_auc", "val_opt"],
                        default="val_opt",
                        help="Weight strategy. val_opt tunes on the val set (recommended).")
    parser.add_argument("--results_dir", default="results/sweep")
    parser.add_argument("--train_dir", default="MINDsmall_train/MINDsmall_train")
    parser.add_argument("--dev_dir",   default="MINDsmall_dev/MINDsmall_dev")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    summary_csv = results_dir / "sweep_summary.csv"

    print(f"\nParsing sweep summary from {summary_csv}...")
    runs = parse_sweep_summary(str(summary_csv))
    runs = [r for r in runs if r["model"] in args.models]
    print(f"  Filtered to {len(runs)} runs for models: {args.models}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    train_news = parse_news(f"{args.train_dir}/news.tsv")
    dev_news   = parse_news(f"{args.dev_dir}/news.tsv")
    all_news   = {**train_news, **dev_news}

    train_behaviors_all = parse_behaviors(f"{args.train_dir}/behaviors.tsv")
    dev_behaviors       = parse_behaviors(f"{args.dev_dir}/behaviors.tsv")

    cats      = sorted({v["category"]    for v in all_news.values()})
    subcats   = sorted({v["subcategory"] for v in all_news.values()})
    cat2idx   = {c: i + 1 for i, c in enumerate(cats)}
    subcat2idx = {s: i + 1 for i, s in enumerate(subcats)}

    all_behaviors = train_behaviors_all + dev_behaviors
    users    = sorted({b["user_id"] for b in all_behaviors})
    user2idx = {u: i + 1 for i, u in enumerate(users)}
    num_users = len(user2idx)

    print("Building vocabulary...")
    vocab = Vocab()
    vocab.build(
        [n["title"] + " " + n["abstract"] for n in all_news.values()],
        min_freq=1,
    )
    print(f"  vocab={len(vocab):,}  users={num_users:,}")

    base_cfg = Config()
    base_cfg.model.num_users = num_users

    # Test set = MINDsmall_dev behaviors
    test_ds = MINDEvalDataset(dev_behaviors, all_news, vocab,
                              user2idx, cat2idx, subcat2idx, base_cfg)
    print(f"  test impressions (MINDsmall_dev): {len(test_ds):,}")

    # Val set = 10% holdout from MINDsmall_train (same split used during training)
    if args.weight_by in ("val_opt", "val_auc"):
        rng = random.Random(SEED)
        idxs = list(range(len(train_behaviors_all)))
        rng.shuffle(idxs)
        n_train = int(len(idxs) * (1.0 - VAL_RATIO))
        train_set = set(idxs[:n_train])
        val_behaviors = [b for i, b in enumerate(train_behaviors_all)
                         if i not in train_set]
        val_ds = MINDEvalDataset(val_behaviors, all_news, vocab,
                                 user2idx, cat2idx, subcat2idx, base_cfg)
        print(f"  val  impressions (MINDsmall_train 10% holdout): {len(val_ds):,}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print(f"\nLoading {len(runs)} model checkpoints...")
    models = []
    for run in runs:
        model = build_model(
            run["model"], len(vocab), len(cat2idx), len(subcat2idx),
            num_users, Config(), hp=run,
        ).to(device)
        load_checkpoint(run["checkpoint"], model, device)
        models.append(model)
        print(f"  {run['model']} (run {int(run['run_id']):03d}): "
              f"val_auc={run.get('best_val_auc', 'n/a')}  "
              f"test_auc={run.get('test_auc', 'n/a')}  ckpt={run['checkpoint']}")

    # ------------------------------------------------------------------
    # Compute weights
    # ------------------------------------------------------------------
    n = len(models)

    if args.weight_by == "uniform":
        weights = np.ones(n, dtype=np.float32) / n
        print("\nWeights: uniform")

    elif args.weight_by == "val_opt":
        print("\nCollecting val-set scores for weight optimisation...")
        per_model_scores, labels_list = collect_scores(
            models, val_ds, device, desc="val scores"
        )
        weights, val_auc = optimize_weights(per_model_scores, labels_list, n)
        print(f"  Val AUC with optimised weights: {val_auc:.4f}")
        print("\nOptimised weights:")
        for run, w in zip(runs, weights):
            print(f"  {run['model']} (run {int(run['run_id']):03d}): {w:.4f}")

    elif args.weight_by == "val_auc":
        raw = np.array([float(r.get("best_val_auc") or 0) for r in runs], dtype=np.float32)
        weights = raw / raw.sum()
        print("\nWeights (proportional to best_val_auc):")
        for run, w in zip(runs, weights):
            print(f"  {run['model']} (run {int(run['run_id']):03d}): {w:.4f}")

    else:  # auc / mrr / ndcg5 — uses test metric from CSV
        col = f"test_{args.weight_by}"
        raw = np.array([float(r[col]) for r in runs], dtype=np.float32)
        weights = raw / raw.sum()
        print(f"\nWeights (proportional to {col}):")
        for run, w in zip(runs, weights):
            print(f"  {run['model']} (run {int(run['run_id']):03d}): {w:.4f}")

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    ensemble = ModelEnsemble(models, weights).to(device)
    print(f"\nEvaluating ensemble on test set ({len(test_ds):,} impressions)...")
    metrics = evaluate(ensemble, test_ds, device)

    best_run = max(runs, key=lambda r: float(r.get("test_auc") or 0))

    print(f"\n{'='*70}")
    print(f"ENSEMBLE RESULTS  (n={n} models, weight_by={args.weight_by})")
    print(f"{'='*70}")
    print(f"  AUC:     {metrics['auc']:.4f}")
    print(f"  MRR:     {metrics['mrr']:.4f}")
    print(f"  nDCG@5:  {metrics['ndcg5']:.4f}")
    print(f"  nDCG@10: {metrics['ndcg10']:.4f}")
    print(f"\nBest individual: {best_run['model']} "
          f"(run {int(best_run['run_id']):03d}, test_auc={float(best_run['test_auc']):.4f})")
    print(f"Ensemble gain:   {metrics['auc'] - float(best_run['test_auc']):+.4f}")


if __name__ == "__main__":
    main()
