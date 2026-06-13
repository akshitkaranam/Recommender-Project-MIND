"""
Entry point – train any MIND model from the command line.

Examples
--------
# NRMS with CUDA (default)
python main.py --model nrms

# NAML, more epochs, smaller batch
python main.py --model naml --epochs 5 --batch_size 32

# LSTUR with "con" fusion, load GloVe
python main.py --model lstur --lstur_mode con --glove path/to/glove.6B.300d.txt

# NPA on CPU
python main.py --model npa --device cpu
"""
import argparse
import random
import sys

import numpy as np
import torch

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.dataset import parse_behaviors, parse_news, MINDTrainDataset, MINDEvalDataset
from data.vocab import Vocab
from models.nrms import NRMS
from models.naml import NAML
from models.lstur import LSTUR
from models.npa import NPA
from trainer import train as run_training


MODEL_REGISTRY = {
    "nrms": NRMS,
    "naml": NAML,
    "lstur": LSTUR,
    "npa": NPA,
}


def parse_args():
    p = argparse.ArgumentParser(description="MIND news recommendation trainer")
    p.add_argument("--model", default="nrms", choices=list(MODEL_REGISTRY))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--neg_samples", type=int, default=4)
    p.add_argument("--max_history", type=int, default=50)
    p.add_argument("--num_heads", type=int, default=20)
    p.add_argument("--head_dim", type=int, default=20)
    p.add_argument("--num_filters", type=int, default=400)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lstur_mode", default="ini", choices=["ini", "con"])
    p.add_argument("--glove", default=None, metavar="PATH")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_dir", default="MINDsmall_train/MINDsmall_train")
    p.add_argument("--dev_dir", default="MINDsmall_dev/MINDsmall_dev")
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    cfg.data.train_dir = args.train_dir
    cfg.data.dev_dir = args.dev_dir
    cfg.data.neg_samples = args.neg_samples
    cfg.data.max_history = args.max_history
    cfg.model.num_heads = args.num_heads
    cfg.model.head_dim = args.head_dim
    cfg.model.num_filters = args.num_filters
    cfg.model.dropout = args.dropout
    cfg.model.lstur_mode = args.lstur_mode
    cfg.train.model_name = args.model
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.device = args.device
    cfg.train.save_dir = args.save_dir
    cfg.train.log_every = args.log_every
    cfg.train.seed = args.seed
    cfg.train.glove_path = args.glove
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    # -----------------------------------------------------------------------
    # Load raw data
    # -----------------------------------------------------------------------
    print("Loading data...")
    train_news = parse_news(f"{cfg.data.train_dir}/news.tsv")
    dev_news = parse_news(f"{cfg.data.dev_dir}/news.tsv")
    all_news = {**train_news, **dev_news}

    train_behaviors = parse_behaviors(f"{cfg.data.train_dir}/behaviors.tsv")
    dev_behaviors = parse_behaviors(f"{cfg.data.dev_dir}/behaviors.tsv")

    # -----------------------------------------------------------------------
    # Build category / user index maps
    # -----------------------------------------------------------------------
    cats = sorted({v["category"] for v in all_news.values()})
    subcats = sorted({v["subcategory"] for v in all_news.values()})
    cat2idx = {c: i + 1 for i, c in enumerate(cats)}
    subcat2idx = {s: i + 1 for i, s in enumerate(subcats)}

    all_behaviors = train_behaviors + dev_behaviors
    users = sorted({b["user_id"] for b in all_behaviors})
    user2idx = {u: i + 1 for i, u in enumerate(users)}
    num_users = len(user2idx)
    cfg.model.num_users = num_users

    # -----------------------------------------------------------------------
    # Build vocabulary
    # -----------------------------------------------------------------------
    print("Building vocabulary...")
    vocab = Vocab()
    texts = [n["title"] + " " + n["abstract"] for n in all_news.values()]
    vocab.build(texts, min_freq=cfg.data.min_word_freq)
    print(f"  Vocab size: {len(vocab):,}  "
          f"Categories: {len(cat2idx)}  Sub-categories: {len(subcat2idx)}  "
          f"Users: {num_users:,}")

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    train_ds = MINDTrainDataset(train_behaviors, all_news, vocab, user2idx, cat2idx, subcat2idx, cfg)
    dev_ds = MINDEvalDataset(dev_behaviors, all_news, vocab, user2idx, cat2idx, subcat2idx, cfg)
    print(f"  Train samples: {len(train_ds):,}  Dev impressions: {len(dev_ds):,}")

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    vocab_size = len(vocab)
    num_cats = len(cat2idx)
    num_subcats = len(subcat2idx)

    if args.model == "nrms":
        model = NRMS(vocab_size, cfg)
    elif args.model == "naml":
        model = NAML(vocab_size, num_cats, num_subcats, cfg)
    elif args.model == "lstur":
        model = LSTUR(vocab_size, num_users, cfg)
    elif args.model == "npa":
        model = NPA(vocab_size, num_users, cfg)
    else:
        sys.exit(f"Unknown model: {args.model}")

    if cfg.train.glove_path:
        pretrained = vocab.load_glove(cfg.train.glove_path, cfg.model.word_emb_dim)
        model.news_enc.emb.weight.data.copy_(pretrained)
        print("  GloVe embeddings loaded.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {args.model.upper()}  Trainable params: {n_params:,}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    run_training(model, train_ds, dev_ds, cfg)


if __name__ == "__main__":
    main()
