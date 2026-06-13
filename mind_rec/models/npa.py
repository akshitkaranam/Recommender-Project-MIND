"""
NPA: Neural News Recommendation with Personalized Attention
Wu et al., 2019  https://arxiv.org/abs/1907.05559
"""
import torch
import torch.nn as nn

from .base import BaseRecommender


class NPANewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_users: int, cfg):
        super().__init__()
        d = cfg.model.word_emb_dim
        f = cfg.model.num_filters
        k = cfg.model.cnn_kernel_size
        uid = cfg.model.user_emb_dim
        uq = cfg.model.user_query_dim

        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, uid, padding_idx=0)
        self.user_proj = nn.Linear(uid, uq)
        self.cnn = nn.Conv1d(d, f, k, padding=k // 2)
        self.word_proj = nn.Linear(f, uq)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, titles: torch.Tensor, user_idx: torch.Tensor = None, **kwargs) -> torch.Tensor:
        mask = titles != 0                                                         # (B, L)
        x = self.dropout(self.emb(titles))
        x = self.dropout(torch.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2)) # (B, L, F)

        if user_idx is not None:
            u_q = torch.relu(self.user_proj(self.user_emb(user_idx)))             # (B, uq)
            attn = (self.word_proj(x) * u_q.unsqueeze(1)).sum(-1)                 # (B, L)
        else:
            attn = x.new_zeros(x.size(0), x.size(1))

        attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).unsqueeze(-1)                          # (B, L, 1)
        return (attn * x).sum(dim=1)                                              # (B, F)


class NPAUserEncoder(nn.Module):
    def __init__(self, num_users: int, news_dim: int, cfg):
        super().__init__()
        uid = cfg.model.user_emb_dim
        uq = cfg.model.user_query_dim
        self.user_emb = nn.Embedding(num_users + 1, uid, padding_idx=0)
        self.user_proj = nn.Linear(uid, uq)
        self.news_proj = nn.Linear(news_dim, uq)

    def forward(self, hist_vecs: torch.Tensor, history_mask: torch.Tensor,
                user_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        u_q = torch.relu(self.user_proj(self.user_emb(user_idx)))                 # (B, uq)
        attn = (self.news_proj(hist_vecs) * u_q.unsqueeze(1)).sum(-1)             # (B, H)
        attn = attn.masked_fill(~history_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).unsqueeze(-1)                          # (B, H, 1)
        return (attn * hist_vecs).sum(dim=1)                                      # (B, D)


class NPA(BaseRecommender):
    """NPA needs the user_idx expanded per-news when encoding, so it overrides forward."""

    def __init__(self, vocab_size: int, num_users: int, cfg):
        super().__init__()
        self.news_enc = NPANewsEncoder(vocab_size, num_users, cfg)
        self.user_enc = NPAUserEncoder(num_users, cfg.model.num_filters, cfg)

    def encode_news(self, titles, abstracts, categories, subcategories, user_idx=None, **kwargs):
        return self.news_enc(titles, user_idx=user_idx)

    def encode_user(self, hist_vecs, history_mask, user_idx=None, **kwargs):
        return self.user_enc(hist_vecs, history_mask, user_idx)

    def forward(self, batch: dict) -> torch.Tensor:
        B, H = batch["history_titles"].shape[:2]
        K = batch["cand_titles"].shape[1]
        user_idx = batch.get("user_idx")   # (B,)

        # broadcast user_idx to match flattened history/candidate batches
        uid_hist = _expand(user_idx, B, H)
        uid_cand = _expand(user_idx, B, K)

        hist_vecs = self.encode_news(
            batch["history_titles"].view(B * H, -1),
            None, None, None,
            user_idx=uid_hist,
        ).view(B, H, -1)

        user_vecs = self.encode_user(hist_vecs, batch["history_mask"], user_idx=user_idx)

        cand_vecs = self.encode_news(
            batch["cand_titles"].view(B * K, -1),
            None, None, None,
            user_idx=uid_cand,
        ).view(B, K, -1)

        return (user_vecs.unsqueeze(1) @ cand_vecs.transpose(1, 2)).squeeze(1)


def _expand(user_idx: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Repeat user_idx N times so it aligns with a (B*N, ...) flattened batch."""
    if user_idx is None:
        return None
    return user_idx.unsqueeze(1).expand(B, N).reshape(B * N)
