from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    train_dir: str = "MINDsmall_train/MINDsmall_train"
    dev_dir: str = "MINDsmall_dev/MINDsmall_dev"
    max_title_len: int = 20
    max_abstract_len: int = 50
    max_history: int = 50
    neg_samples: int = 4
    min_word_freq: int = 1


@dataclass
class ModelConfig:
    word_emb_dim: int = 300
    num_heads: int = 20
    head_dim: int = 20          # news_dim = num_heads * head_dim = 400
    query_dim: int = 200
    dropout: float = 0.2
    # NAML / LSTUR / NPA
    cnn_kernel_size: int = 3
    num_filters: int = 400
    # LSTUR
    lstur_mode: str = "ini"     # "ini" or "con"
    # NPA / LSTUR
    num_users: int = 0          # filled at runtime from data
    user_emb_dim: int = 50
    user_query_dim: int = 200


@dataclass
class TrainConfig:
    model_name: str = "nrms"    # nrms | naml | lstur | npa
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda"
    save_dir: str = "checkpoints"
    log_every: int = 100
    eval_every: int = 1
    seed: int = 42
    glove_path: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
