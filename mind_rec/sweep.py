#!/usr/bin/env python3
"""
Hyperparameter sweep for all MIND recommendation models.

Splits train behaviors 90/10 into train/val. Tunes on val loss (CrossEntropyLoss)
with early stopping (patience=5). Reports final metrics on the held-out dev set.

Usage (from mind_rec/ directory):
    python sweep.py                           # all 4 models, all configs
    python sweep.py --models nrms naml        # subset of models
    python sweep.py --device cpu              # force CPU
    python sweep.py --val_ratio 0.15          # 15% val split
    python sweep.py --results_dir my_results  # custom output dir

Outputs
-------
results/sweep/
    epoch_metrics.csv   - one row per (run, epoch)
    sweep_summary.csv   - one row per run with best val + test metrics
    checkpoints/        - best .pt per run
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import parse_behaviors, parse_news, MINDTrainDataset, MINDEvalDataset
from data.vocab import Vocab
from evaluate import compute_metrics
from models.nrms import NRMS
from models.naml import NAML
from models.lstur import LSTUR
from models.npa import NPA

# ---------------------------------------------------------------------------
# Global training constants
# ---------------------------------------------------------------------------
MAX_EPOCHS = 30   # hard ceiling; early stopping should fire well before this
PATIENCE   = 3    # stop when val_loss has not improved for this many epochs
SEED       = 42

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------
# Each dict lists the hyperparameters that differ across configs for that model.
# Keys not present fall back to defaults set in build_cfg().
#
# NRMS  : news_dim = num_heads * head_dim  (always 400 here)
# NAML  : news_dim = num_filters
# LSTUR : news_dim = num_filters
# NPA   : news_dim = num_filters
# ---------------------------------------------------------------------------
SWEEP_CONFIGS: Dict[str, List[dict]] = {
    "nrms": [
        # baseline
        {"lr": 1e-4, "dropout": 0.2, "num_heads": 20, "head_dim": 20, "batch_size": 64},
        # higher lr
        {"lr": 3e-4, "dropout": 0.2, "num_heads": 20, "head_dim": 20, "batch_size": 64},
        {"lr": 1e-3, "dropout": 0.2, "num_heads": 20, "head_dim": 20, "batch_size": 64},
        # higher dropout
        {"lr": 1e-4, "dropout": 0.3, "num_heads": 20, "head_dim": 20, "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.3, "num_heads": 20, "head_dim": 20, "batch_size": 64},
        # larger batch
        {"lr": 1e-4, "dropout": 0.1, "num_heads": 20, "head_dim": 20, "batch_size": 128},
        {"lr": 3e-4, "dropout": 0.1, "num_heads": 20, "head_dim": 20, "batch_size": 128},
        # different head factorisation (news_dim still 400)
        {"lr": 1e-4, "dropout": 0.2, "num_heads": 16, "head_dim": 25, "batch_size": 64},
    ],
    "naml": [
        {"lr": 1e-4, "dropout": 0.2, "num_filters": 400, "cnn_kernel_size": 3, "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.2, "num_filters": 400, "cnn_kernel_size": 3, "batch_size": 64},
        {"lr": 1e-4, "dropout": 0.3, "num_filters": 400, "cnn_kernel_size": 3, "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.2, "num_filters": 400, "cnn_kernel_size": 5, "batch_size": 64},
    ],
    "lstur": [
        # covers both fusion modes × two lr values — mode is the key LSTUR HP
        {"lr": 1e-4, "dropout": 0.2, "num_filters": 400, "lstur_mode": "ini", "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.2, "num_filters": 400, "lstur_mode": "ini", "batch_size": 64},
        {"lr": 1e-4, "dropout": 0.2, "num_filters": 400, "lstur_mode": "con", "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.2, "num_filters": 400, "lstur_mode": "con", "batch_size": 64},
    ],
    "npa": [
        {"lr": 1e-4, "dropout": 0.2, "num_filters": 400, "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.2, "num_filters": 400, "batch_size": 64},
        {"lr": 1e-4, "dropout": 0.3, "num_filters": 400, "batch_size": 64},
        {"lr": 3e-4, "dropout": 0.1, "num_filters": 400, "batch_size": 128},
    ],
}

# All HP keys that ever appear across all models (used for uniform CSV columns)
ALL_HP_KEYS = [
    "lr", "dropout", "batch_size",
    "num_heads", "head_dim",
    "num_filters", "cnn_kernel_size",
    "lstur_mode",
]

EPOCH_FIELDS = (
    ["run_id", "model"] + ALL_HP_KEYS
    + ["epoch", "train_loss", "val_loss",
       "val_auc", "val_mrr", "val_ndcg5", "val_ndcg10", "is_best"]
)

SUMMARY_FIELDS = (
    ["run_id", "model"] + ALL_HP_KEYS
    + ["best_epoch", "best_val_loss",
       "best_val_auc", "best_val_mrr", "best_val_ndcg5", "best_val_ndcg10",
       "test_auc", "test_mrr", "test_ndcg5", "test_ndcg10",
       "n_params", "checkpoint"]
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_behaviors(
    behaviors: List[dict],
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> tuple[List[dict], List[dict]]:
    """Randomly partition behaviors into (train, val) without modifying the original."""
    rng = random.Random(seed)
    idxs = list(range(len(behaviors)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * (1.0 - val_ratio))
    train_set = set(idxs[:n_train])
    train_b = [b for i, b in enumerate(behaviors) if i in train_set]
    val_b   = [b for i, b in enumerate(behaviors) if i not in train_set]
    return train_b, val_b


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def build_cfg(hp: dict, model_name: str, device: str, save_dir: str) -> Config:
    cfg = Config()
    cfg.train.model_name      = model_name
    cfg.train.lr              = hp.get("lr", 1e-4)
    cfg.train.batch_size      = hp.get("batch_size", 64)
    cfg.train.device          = device
    cfg.train.save_dir        = save_dir
    cfg.train.seed            = SEED
    cfg.train.epochs          = MAX_EPOCHS
    cfg.model.dropout         = hp.get("dropout", 0.2)
    cfg.model.num_heads       = hp.get("num_heads", 20)
    cfg.model.head_dim        = hp.get("head_dim", 20)
    cfg.model.num_filters     = hp.get("num_filters", 400)
    cfg.model.cnn_kernel_size = hp.get("cnn_kernel_size", 3)
    cfg.model.lstur_mode      = hp.get("lstur_mode", "ini")
    return cfg


def build_model(
    model_name: str,
    vocab_size: int,
    num_cats: int,
    num_subcats: int,
    num_users: int,
    cfg: Config,
) -> nn.Module:
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


def compute_val_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int = None,
) -> float:
    """Batched val loss on MINDTrainDataset samples — fast, deterministic early-stopping signal."""
    use_amp = device.type == "cuda"
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                total += criterion(model(batch), batch["label"]).item()
            n += 1
            if max_steps and n >= max_steps:
                break
    return total / max(n, 1)


def compute_eval_metrics(
    model: nn.Module,
    eval_ds,
    device: torch.device,
) -> dict:
    """Full ranking metrics (AUC, MRR, nDCG) on MINDEvalDataset."""
    use_amp = device.type == "cuda"
    model.eval()
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                scores = model(batch).squeeze(0).cpu().numpy()
            labels = batch["labels"].squeeze(0).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)
    return compute_metrics(all_scores, all_labels)


def free_gpu(model: nn.Module) -> None:
    """Delete model and flush GPU memory allocator."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Single training run with early stopping
# ---------------------------------------------------------------------------

def run_single(
    run_id: int,
    model_name: str,
    hp: dict,
    shared: dict,
    device_str: str,
    ckpt_dir: str,
    epoch_writer: csv.DictWriter,
    epoch_file,
    summary_writer: csv.DictWriter,
    summary_file,
    *,
    max_epochs: int = MAX_EPOCHS,
    skip_epoch_auc: bool = False,
    num_workers: int = 0,
    max_steps_per_epoch: int = None,
    use_compile: bool = False,
) -> dict:
    print(f"\n{'='*72}")
    print(f"  Run {run_id:03d} | {model_name.upper()} | {hp}")
    print(f"{'='*72}")

    set_seed(SEED)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    cfg    = build_cfg(hp, model_name, device_str, ckpt_dir)

    model = build_model(
        model_name,
        shared["vocab_size"],
        shared["num_cats"],
        shared["num_subcats"],
        shared["num_users"],
        cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}   Device: {device}")

    # torch.compile requires MSVC (cl.exe) on Windows; pass --compile to enable
    if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        if sys.platform == "win32":
            import shutil, glob
            if shutil.which("cl") is None:
                # Auto-locate cl.exe under VS 2019/2022 installations
                patterns = [
                    r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
                    r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
                ]
                cl_dir = None
                for pat in patterns:
                    hits = glob.glob(pat)
                    if hits:
                        cl_dir = sorted(hits)[-1]   # latest version
                        break
                if cl_dir:
                    os.environ["PATH"] = cl_dir + os.pathsep + os.environ["PATH"]
                    print(f"  cl.exe auto-added to PATH: {cl_dir}")
                else:
                    print("  torch.compile: skipped (cl.exe not found — install VS Build Tools with C++ workload)")
                    use_compile = False
        if use_compile:
            model = torch.compile(model, dynamic=True)
            print("  torch.compile: enabled")

    train_loader = DataLoader(
        shared["train_ds"],
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loss_loader = DataLoader(
        shared["val_train_ds"],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_run{run_id:03d}_best.pt")

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Early-stopping state
    best_val_loss    = float("inf")
    patience_counter = 0
    best_epoch       = 0
    best_val_metrics: dict = {}

    for epoch in range(1, max_epochs + 1):

        # ---- Training pass (AMP when on CUDA) ------------------------------
        model.train()
        total_loss, n_steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  [Ep {epoch:02d}] train", leave=False)
        for batch in pbar:
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                scores = model(batch)
                loss   = criterion(scores, batch["label"])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix(loss=f"{total_loss / n_steps:.4f}")
            if max_steps_per_epoch and n_steps >= max_steps_per_epoch:
                break
        train_loss = total_loss / max(n_steps, 1)

        # ---- Val loss (early-stopping signal) ------------------------------
        val_loss = compute_val_loss(
            model, val_loss_loader, criterion, device,
            max_steps=max_steps_per_epoch,
        )

        # ---- Val ranking metrics (skipped per-epoch when --skip_epoch_auc) --
        if not skip_epoch_auc:
            val_metrics = compute_eval_metrics(model, shared["val_eval_ds"], device)
        else:
            val_metrics = {}

        # ---- Checkpoint & early stopping -----------------------------------
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss    = val_loss
            best_epoch       = epoch
            if not skip_epoch_auc:
                best_val_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = " ← best"
        else:
            patience_counter += 1
            flag = f"  (no improvement {patience_counter}/{PATIENCE})"

        if skip_epoch_auc:
            print(
                f"  Ep {epoch:02d}"
                f"  train={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"{flag}"
            )
        else:
            print(
                f"  Ep {epoch:02d}"
                f"  train={train_loss:.4f}"
                f"  val_loss={val_loss:.4f}"
                f"  auc={val_metrics['auc']:.4f}"
                f"  mrr={val_metrics['mrr']:.4f}"
                f"  ndcg5={val_metrics['ndcg5']:.4f}"
                f"  ndcg10={val_metrics['ndcg10']:.4f}"
                f"{flag}"
            )

        # Write per-epoch CSV row immediately
        row = {"run_id": run_id, "model": model_name}
        row.update({k: hp.get(k, "") for k in ALL_HP_KEYS})
        row.update({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_auc":    round(val_metrics["auc"], 6) if val_metrics else "",
            "val_mrr":    round(val_metrics["mrr"], 6) if val_metrics else "",
            "val_ndcg5":  round(val_metrics["ndcg5"], 6) if val_metrics else "",
            "val_ndcg10": round(val_metrics["ndcg10"], 6) if val_metrics else "",
            "is_best":    int(is_best),
        })
        epoch_writer.writerow(row)
        epoch_file.flush()

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}  (best was epoch {best_epoch}).")
            break

    # ---- Load best checkpoint; compute val AUC if it was skipped ----------
    if best_epoch == 0 or not os.path.exists(ckpt_path):
        print("\n  Warning: no checkpoint saved (training may have diverged). Using current model state.")
    else:
        print(f"\n  Loading epoch-{best_epoch} checkpoint for final evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    if skip_epoch_auc and not max_steps_per_epoch:
        print("  Computing val AUC from best checkpoint...")
        best_val_metrics = compute_eval_metrics(model, shared["val_eval_ds"], device)

    if max_steps_per_epoch:
        print("  Smoke test — skipping full val/test AUC eval.")
        test_metrics = {"auc": float("nan"), "mrr": float("nan"),
                        "ndcg5": float("nan"), "ndcg10": float("nan")}
    else:
        test_metrics = compute_eval_metrics(model, shared["test_ds"], device)

    print(
        f"  TEST → auc={test_metrics['auc']:.4f}"
        f"  mrr={test_metrics['mrr']:.4f}"
        f"  ndcg5={test_metrics['ndcg5']:.4f}"
        f"  ndcg10={test_metrics['ndcg10']:.4f}"
    )

    # Write summary CSV row
    summary_row = {"run_id": run_id, "model": model_name}
    summary_row.update({k: hp.get(k, "") for k in ALL_HP_KEYS})
    summary_row.update({
        "best_epoch":     best_epoch,
        "best_val_loss":  round(best_val_loss, 6),
        "best_val_auc":   round(best_val_metrics.get("auc", float("nan")), 6),
        "best_val_mrr":   round(best_val_metrics.get("mrr", float("nan")), 6),
        "best_val_ndcg5": round(best_val_metrics.get("ndcg5", float("nan")), 6),
        "best_val_ndcg10":round(best_val_metrics.get("ndcg10", float("nan")), 6),
        "test_auc":       round(test_metrics["auc"], 6),
        "test_mrr":       round(test_metrics["mrr"], 6),
        "test_ndcg5":     round(test_metrics["ndcg5"], 6),
        "test_ndcg10":    round(test_metrics["ndcg10"], 6),
        "n_params":       n_params,
        "checkpoint":     ckpt_path,
    })
    summary_writer.writerow(summary_row)
    summary_file.flush()

    # ---- GPU cleanup -------------------------------------------------------
    free_gpu(model)
    print("  GPU memory cleared.")

    return summary_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIND HP sweep — all models")
    p.add_argument(
        "--models", nargs="+", default=list(SWEEP_CONFIGS),
        choices=list(SWEEP_CONFIGS),
        help="Models to sweep (default: all four)",
    )
    p.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of train behaviors reserved for validation (default: 0.1)",
    )
    p.add_argument("--device", default="cuda",
                   help="Compute device; falls back to cpu if CUDA unavailable")
    p.add_argument("--train_dir", default="MINDsmall_train/MINDsmall_train")
    p.add_argument("--dev_dir",   default="MINDsmall_dev/MINDsmall_dev")
    p.add_argument(
        "--results_dir", default="results/sweep",
        help="Root for CSVs and checkpoints (default: results/sweep)",
    )
    # ---- Speed options -------------------------------------------------------
    p.add_argument(
        "--subsample_train", type=float, default=1.0, metavar="FRAC",
        help="Fraction of train behaviors to use (e.g. 0.2). "
             "Val set is always kept full. Huge speedup for HP discovery. (default: 1.0)",
    )
    p.add_argument(
        "--skip_epoch_auc", action="store_true",
        help="Skip the slow per-epoch val AUC eval; compute it once from the best "
             "checkpoint at the end of each run. Saves ~30-40%% per epoch.",
    )
    p.add_argument(
        "--max_epochs", type=int, default=MAX_EPOCHS,
        help=f"Hard epoch ceiling per run (default: {MAX_EPOCHS}). "
             "Early stopping will usually fire before this.",
    )
    p.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes for async data prefetch. "
             "0 = main process only (safe on Windows). (default: 0)",
    )
    p.add_argument(
        "--compile", action="store_true",
        help="Enable torch.compile (requires MSVC cl.exe on Windows).",
    )
    p.add_argument(
        "--smoke_test", action="store_true",
        help="Quick pipeline check: 1st config per model, 1 epoch, 5 gradient steps.",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(SEED)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # TF32 is faster than FP32 on Ampere+ GPUs with negligible precision loss
        torch.set_float32_matmul_precision("high")
        print("  TF32 matmul precision enabled.")
    else:
        print("No CUDA GPU found — running on CPU.")

    results_dir = Path(args.results_dir)
    ckpt_dir    = str(results_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load raw data once — shared across all runs
    # -----------------------------------------------------------------------
    print("\nLoading news and behavior files...")
    train_news = parse_news(f"{args.train_dir}/news.tsv")
    dev_news   = parse_news(f"{args.dev_dir}/news.tsv")
    all_news   = {**train_news, **dev_news}

    train_behaviors_all = parse_behaviors(f"{args.train_dir}/behaviors.tsv")
    dev_behaviors       = parse_behaviors(f"{args.dev_dir}/behaviors.tsv")

    # -----------------------------------------------------------------------
    # Train / val split (deterministic; same for every run in this sweep)
    # -----------------------------------------------------------------------
    train_behaviors, val_behaviors = split_behaviors(
        train_behaviors_all, val_ratio=args.val_ratio, seed=SEED
    )
    print(
        f"  Behavior split — train: {len(train_behaviors):,}"
        f"  val: {len(val_behaviors):,}"
        f"  test (dev): {len(dev_behaviors):,}"
    )

    # ---- Optionally subsample training behaviors for faster HP sweeps ------
    # Val is always kept full so early-stopping signal stays reliable.
    if args.subsample_train < 1.0:
        rng = random.Random(SEED)
        n_sub = max(1, int(len(train_behaviors) * args.subsample_train))
        train_behaviors = rng.sample(train_behaviors, n_sub)
        print(f"  Train subsampled to {n_sub:,} behaviors ({args.subsample_train*100:.0f}% of full train split)")

    # -----------------------------------------------------------------------
    # Build index maps — shared across all runs
    # -----------------------------------------------------------------------
    cats       = sorted({v["category"]    for v in all_news.values()})
    subcats    = sorted({v["subcategory"] for v in all_news.values()})
    cat2idx    = {c: i + 1 for i, c in enumerate(cats)}
    subcat2idx = {s: i + 1 for i, s in enumerate(subcats)}

    all_behaviors = train_behaviors_all + dev_behaviors
    users    = sorted({b["user_id"] for b in all_behaviors})
    user2idx = {u: i + 1 for i, u in enumerate(users)}
    num_users = len(user2idx)

    # -----------------------------------------------------------------------
    # Vocabulary — built once from all news text
    # -----------------------------------------------------------------------
    print("Building vocabulary...")
    vocab = Vocab()
    vocab.build(
        [n["title"] + " " + n["abstract"] for n in all_news.values()],
        min_freq=1,
    )
    print(
        f"  vocab={len(vocab):,}  cats={len(cat2idx)}"
        f"  subcats={len(subcat2idx)}  users={num_users:,}"
    )

    # -----------------------------------------------------------------------
    # Datasets — built once; DataLoader (with batch_size) created per run
    #
    # MINDTrainDataset samples negatives at __init__ time, so train_ds and
    # val_train_ds have fixed (pos, neg) pairs — val_loss is deterministic
    # across epochs, giving a stable early-stopping signal.
    # -----------------------------------------------------------------------
    print("Building datasets...")
    base_cfg = Config()          # default DataConfig; only cfg.data.* is used
    base_cfg.model.num_users = num_users

    train_ds     = MINDTrainDataset(train_behaviors, all_news, vocab,
                                    user2idx, cat2idx, subcat2idx, base_cfg)
    val_train_ds = MINDTrainDataset(val_behaviors,   all_news, vocab,
                                    user2idx, cat2idx, subcat2idx, base_cfg)
    val_eval_ds  = MINDEvalDataset(val_behaviors,    all_news, vocab,
                                   user2idx, cat2idx, subcat2idx, base_cfg)
    test_ds      = MINDEvalDataset(dev_behaviors,    all_news, vocab,
                                   user2idx, cat2idx, subcat2idx, base_cfg)

    print(
        f"  train samples: {len(train_ds):,}"
        f"  val train samples: {len(val_train_ds):,}"
        f"  val eval impressions: {len(val_eval_ds):,}"
        f"  test impressions: {len(test_ds):,}"
    )

    shared = {
        "vocab_size":  len(vocab),
        "num_cats":    len(cat2idx),
        "num_subcats": len(subcat2idx),
        "num_users":   num_users,
        "train_ds":    train_ds,
        "val_train_ds": val_train_ds,
        "val_eval_ds": val_eval_ds,
        "test_ds":     test_ds,
    }

    # -----------------------------------------------------------------------
    # Open result CSVs (written incrementally so partial sweeps are not lost)
    # -----------------------------------------------------------------------
    epoch_csv_path   = results_dir / "epoch_metrics.csv"
    summary_csv_path = results_dir / "sweep_summary.csv"

    epoch_file   = open(epoch_csv_path,   "w", newline="", encoding="utf-8")
    summary_file = open(summary_csv_path, "w", newline="", encoding="utf-8")

    epoch_writer   = csv.DictWriter(epoch_file,   fieldnames=EPOCH_FIELDS,
                                    extrasaction="ignore", restval="")
    summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS,
                                    extrasaction="ignore", restval="")
    epoch_writer.writeheader()
    summary_writer.writeheader()

    # -----------------------------------------------------------------------
    # Sweep loop
    # -----------------------------------------------------------------------
    run_id       = 0
    all_summaries: List[dict] = []
    total_runs   = sum(len(SWEEP_CONFIGS[m]) for m in args.models)
    t0           = time.time()

    for model_name in args.models:
        configs = SWEEP_CONFIGS[model_name]
        print(f"\n{'#'*72}")
        print(f"  {model_name.upper()} sweep — {len(configs)} configs")
        print(f"{'#'*72}")

        if args.smoke_test:
            configs = configs[:1]

        for hp in configs:
            run_id += 1
            print(f"\n  [{run_id}/{total_runs}]  elapsed: {(time.time()-t0)/60:.1f} min")
            summary = run_single(
                run_id, model_name, hp, shared,
                device, ckpt_dir,
                epoch_writer, epoch_file,
                summary_writer, summary_file,
                max_epochs=1 if args.smoke_test else args.max_epochs,
                skip_epoch_auc=True if args.smoke_test else args.skip_epoch_auc,
                num_workers=args.num_workers,
                max_steps_per_epoch=5 if args.smoke_test else None,
                use_compile=args.compile,
            )
            all_summaries.append(summary)

    epoch_file.close()
    summary_file.close()

    # -----------------------------------------------------------------------
    # Final leaderboard
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("SWEEP COMPLETE — Best run per model (by test AUC)")
    print(f"{'='*72}\n")

    by_model: Dict[str, List[dict]] = {}
    for s in all_summaries:
        by_model.setdefault(s["model"], []).append(s)

    for mname, runs in by_model.items():
        best = max(runs, key=lambda r: float(r.get("test_auc") or 0))
        active_hps = {k: best[k] for k in ALL_HP_KEYS if best.get(k) not in ("", None)}
        print(f"  {mname.upper()}  (run {best['run_id']:03d})")
        print(f"    HPs      : {active_hps}")
        print(f"    Best epoch: {best['best_epoch']}")
        print(
            f"    Val  → auc={best['best_val_auc']}  mrr={best['best_val_mrr']}"
            f"  ndcg5={best['best_val_ndcg5']}  ndcg10={best['best_val_ndcg10']}"
        )
        print(
            f"    Test → auc={best['test_auc']}  mrr={best['test_mrr']}"
            f"  ndcg5={best['test_ndcg5']}  ndcg10={best['test_ndcg10']}"
        )
        print(f"    Ckpt : {best['checkpoint']}\n")

    print(f"Results written to:")
    print(f"  {epoch_csv_path}")
    print(f"  {summary_csv_path}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
