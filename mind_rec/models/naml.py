"""
NAML: Neural News Recommendation with Attentive Multi-View Learning
Wu et al., 2019  https://arxiv.org/abs/1907.05576
"""
import torch
import torch.nn as nn

from .base import AdditiveAttention, BaseRecommender


class TextCNNEncoder(nn.Module):
    """Shared word-embedding CNN + additive attention for one text field."""

    def __init__(self, shared_emb: nn.Embedding, num_filters: int, kernel_size: int,
                 query_dim: int, dropout: float):
        super().__init__()
        self.emb = shared_emb
        d = shared_emb.embedding_dim
        self.cnn = nn.Conv1d(d, num_filters, kernel_size, padding=kernel_size // 2)
        self.attn = AdditiveAttention(num_filters, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0
        e = self.dropout(self.emb(x))                             # (B, L, D)
        c = torch.relu(self.cnn(e.transpose(1, 2))).transpose(1, 2)  # (B, L, F)
        return self.attn(self.dropout(c), mask)                   # (B, F)


class NAMLNewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_categories: int, num_subcategories: int, cfg):
        super().__init__()
        d = cfg.model.word_emb_dim
        f = cfg.model.num_filters
        k = cfg.model.cnn_kernel_size
        q = cfg.model.query_dim
        dr = cfg.model.dropout

        shared_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.title_enc = TextCNNEncoder(shared_emb, f, k, q, dr)
        self.abstract_enc = TextCNNEncoder(shared_emb, f, k, q, dr)

        # category & subcategory projected to same dim as CNN output
        self.cat_emb = nn.Embedding(num_categories + 1, f, padding_idx=0)
        self.subcat_emb = nn.Embedding(num_subcategories + 1, f, padding_idx=0)

        self.view_attn = AdditiveAttention(f, q)

    def forward(self, titles, abstracts, categories, subcategories, **kwargs):
        t = self.title_enc(titles)                    # (B, F)
        a = self.abstract_enc(abstracts)              # (B, F)
        c = self.cat_emb(categories)                 # (B, F)
        s = self.subcat_emb(subcategories)            # (B, F)
        views = torch.stack([t, a, c, s], dim=1)     # (B, 4, F)
        return self.view_attn(views)                  # (B, F)


class NAMLUserEncoder(nn.Module):
    def __init__(self, news_dim: int, query_dim: int):
        super().__init__()
        self.attn = AdditiveAttention(news_dim, query_dim)

    def forward(self, hist_vecs, history_mask, **kwargs):
        return self.attn(hist_vecs, history_mask)


class NAML(BaseRecommender):
    def __init__(self, vocab_size: int, num_categories: int, num_subcategories: int, cfg):
        super().__init__()
        self.news_enc = NAMLNewsEncoder(vocab_size, num_categories, num_subcategories, cfg)
        self.user_enc = NAMLUserEncoder(cfg.model.num_filters, cfg.model.query_dim)

    def encode_news(self, titles, abstracts, categories, subcategories, **kwargs):
        return self.news_enc(titles, abstracts, categories, subcategories)

    def encode_user(self, hist_vecs, history_mask, **kwargs):
        return self.user_enc(hist_vecs, history_mask)
