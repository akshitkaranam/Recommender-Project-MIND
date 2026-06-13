import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import compute_metrics


def train(model: nn.Module, train_dataset, dev_dataset, cfg) -> nn.Module:
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}")
        for step, batch in enumerate(pbar, 1):
            batch = _to_device(batch, device)
            scores = model(batch)                          # (B, K)
            loss = criterion(scores, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}  avg_loss={avg_loss:.4f}")

        if epoch % cfg.train.eval_every == 0:
            metrics = evaluate(model, dev_dataset, cfg, device)
            print(
                f"  AUC={metrics['auc']:.4f}  MRR={metrics['mrr']:.4f}"
                f"  nDCG@5={metrics['ndcg5']:.4f}  nDCG@10={metrics['ndcg10']:.4f}"
                f"  ({metrics['n_impressions']} impressions)"
            )
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                path = os.path.join(cfg.train.save_dir, f"{cfg.train.model_name}_best.pt")
                torch.save(model.state_dict(), path)
                print(f"  Saved → {path}")

    return model


def evaluate(model: nn.Module, dataset, cfg, device=None) -> dict:
    if device is None:
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model.eval()

    # batch_size=1 because each impression has a different number of candidates
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            scores = model(batch).squeeze(0).cpu().numpy()
            labels = batch["labels"].squeeze(0).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)

    return compute_metrics(all_scores, all_labels)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
