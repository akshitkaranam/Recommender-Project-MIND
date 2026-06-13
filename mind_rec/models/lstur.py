"""
LSTUR: Neural News Recommendation with Long- and Short-term User Representations
An et al., 2019  https://arxiv.org/abs/1905.13226
"""
import torch
import torch.nn as nn

from .base import AdditiveAttention, BaseRecommender


class LSTURNewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        d = cfg.model.word_emb_dim
        f = cfg.model.num_filters
        k = cfg.model.cnn_kernel_size
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.cnn = nn.Conv1d(d, f, k, padding=k // 2)
        self.attn = AdditiveAttention(f, cfg.model.query_dim)
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, titles: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = titles != 0
        x = self.dropout(self.emb(titles))
        x = self.dropout(torch.relu(self.cnn(x.transpose(1, 2))).transpose(1, 2))
        return self.attn(x, mask)


class LSTURUserEncoder(nn.Module):
    """
    mode="ini": long-term user embedding initialises the GRU hidden state.
    mode="con": GRU short-term output is concatenated with long-term embedding.
    """

    def __init__(self, num_users: int, news_dim: int, cfg):
        super().__init__()
        self.mode = cfg.model.lstur_mode
        uid = cfg.model.user_emb_dim
        self.user_emb = nn.Embedding(num_users + 1, uid, padding_idx=0)

        if self.mode == "ini":
            self.ini_proj = nn.Linear(uid, news_dim)
            self.gru = nn.GRU(news_dim, news_dim, batch_first=True)
        else:
            self.gru = nn.GRU(news_dim, news_dim, batch_first=True)
            self.con_proj = nn.Linear(news_dim + uid, news_dim)

    def forward(self, hist_vecs: torch.Tensor, history_mask: torch.Tensor,
                user_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        long_term = self.user_emb(user_idx)            # (B, uid)

        if self.mode == "ini":
            h0 = torch.tanh(self.ini_proj(long_term)).unsqueeze(0)   # (1, B, news_dim)
            out, _ = self.gru(hist_vecs, h0)           # (B, H, news_dim)
        else:
            out, _ = self.gru(hist_vecs)               # (B, H, news_dim)

        # select the last valid (non-padding) hidden state as short-term rep
        lengths = history_mask.long().sum(dim=1).clamp(min=1) - 1    # (B,)
        short_term = out[torch.arange(out.size(0), device=out.device), lengths]  # (B, news_dim)

        if self.mode == "con":
            short_term = torch.relu(self.con_proj(torch.cat([short_term, long_term], dim=-1)))

        return short_term


class LSTUR(BaseRecommender):
    def __init__(self, vocab_size: int, num_users: int, cfg):
        super().__init__()
        self.news_enc = LSTURNewsEncoder(vocab_size, cfg)
        self.user_enc = LSTURUserEncoder(num_users, cfg.model.num_filters, cfg)

    def encode_news(self, titles, abstracts, categories, subcategories, **kwargs):
        return self.news_enc(titles)

    def encode_user(self, hist_vecs, history_mask, user_idx=None, **kwargs):
        return self.user_enc(hist_vecs, history_mask, user_idx)
