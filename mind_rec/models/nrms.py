"""
NRMS: Neural News Recommendation with Multi-Head Self-Attention
Wu et al., 2019  https://aclanthology.org/D19-1671/
"""
import torch
import torch.nn as nn

from .base import AdditiveAttention, BaseRecommender


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        out_dim = num_heads * head_dim
        self.Wq = nn.Linear(in_dim, out_dim)
        self.Wk = nn.Linear(in_dim, out_dim)
        self.Wv = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, hd = self.num_heads, self.head_dim

        def split(t):
            return t.view(B, L, H, hd).transpose(1, 2)   # (B, H, L, hd)

        Q, K, V = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        attn = self.dropout(torch.softmax((Q @ K.transpose(-1, -2)) / self.scale, dim=-1))
        return (attn @ V).transpose(1, 2).contiguous().view(B, L, H * hd)


def _encode_text(emb, mhsa, attn, dropout, tokens):
    """Shared helper: embed → MHSA → additive attention → vector."""
    mask = tokens != 0
    x = mhsa(dropout(emb(tokens)))
    return attn(x, mask)


class NRMSNewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        d, h = cfg.model.word_emb_dim, cfg.model.num_heads * cfg.model.head_dim
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.dropout = nn.Dropout(cfg.model.dropout)

        # separate MHSA + attention for title and abstract
        self.title_mhsa = MultiHeadSelfAttention(d, cfg.model.num_heads, cfg.model.head_dim, cfg.model.dropout)
        self.title_attn = AdditiveAttention(h, cfg.model.query_dim)

        self.abstract_mhsa = MultiHeadSelfAttention(d, cfg.model.num_heads, cfg.model.head_dim, cfg.model.dropout)
        self.abstract_attn = AdditiveAttention(h, cfg.model.query_dim)

        # fuse two views into one news vector
        self.view_attn = AdditiveAttention(h, cfg.model.query_dim)

    def forward(self, titles: torch.Tensor, abstracts: torch.Tensor, **kwargs) -> torch.Tensor:
        title_vec = _encode_text(self.emb, self.title_mhsa, self.title_attn, self.dropout, titles)
        abstract_vec = _encode_text(self.emb, self.abstract_mhsa, self.abstract_attn, self.dropout, abstracts)
        views = torch.stack([title_vec, abstract_vec], dim=1)   # (B, 2, h)
        return self.view_attn(views)                            # (B, h)


class NRMSUserEncoder(nn.Module):
    def __init__(self, news_dim: int, cfg):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(news_dim, cfg.model.num_heads, cfg.model.head_dim, cfg.model.dropout)
        self.attn = AdditiveAttention(news_dim, cfg.model.query_dim)

    def forward(self, hist_vecs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        x = self.mhsa(hist_vecs)
        return self.attn(x, history_mask)


class NRMS(BaseRecommender):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.news_enc = NRMSNewsEncoder(vocab_size, cfg)
        self.user_enc = NRMSUserEncoder(cfg.model.num_heads * cfg.model.head_dim, cfg)

    def encode_news(self, titles, abstracts, categories, subcategories, **kwargs):
        return self.news_enc(titles, abstracts)

    def encode_user(self, hist_vecs, history_mask, **kwargs):
        return self.user_enc(hist_vecs, history_mask)
