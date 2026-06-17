"""
UniUM-Fastformer adapted from Donmaxo/fastformer-for-rec-UofG (PyTorch).

Key differences vs. our original implementation:
- FastSelfAttention uses Linear(hidden, num_heads) per-head scalar scoring
  (not per-head weight vectors), plus a transform + Q residual at the end.
- FastAttention wraps that with BertSelfOutput (Linear + Dropout + LayerNorm).
- FastformerLayer adds a full FFN block (BertIntermediate + BertOutput style).
- FastformerEncoder stacks N layers over position embeddings, then pools with
  AttentionPooling (exp-weighted, mask-aware).
- Both news and user encoders use FastformerEncoder (news via word embeddings
  since we have no PLM; user via history news vectors — matching their UserEncoder).
news_dim = num_heads * head_dim, same contract as NRMS.
"""
import torch
import torch.nn as nn

from .base import BaseRecommender


# ── building blocks ────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Exp-weighted attention pooling matching their implementation.
    alpha = exp(fc2(tanh(fc1(x)))) * mask, then normalise.
    """

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)  mask: (B, L) float 1=valid 0=pad
        B = x.shape[0]
        alpha = self.fc2(torch.tanh(self.fc1(x)))          # (B, L, 1)
        alpha = torch.exp(alpha)
        if mask is not None:
            alpha = alpha * mask.unsqueeze(2)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        out = torch.bmm(x.permute(0, 2, 1), alpha)         # (B, D, 1)
        return out.reshape(B, -1)                           # (B, D)


class FastSelfAttention(nn.Module):
    """
    Exact match of FastSelfAttention from the reference repo.
    Two-pass additive pooling:
      1. Pool Q → global query q̄  (scored by query_att)
      2. R = K * q̄ (element-wise)
      3. Pool R → global key k̄    (scored by key_att)
      4. out = transform(k̄ * Q_heads) + Q  (residual to Q)
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query     = nn.Linear(hidden_size, hidden_size)
        self.query_att = nn.Linear(hidden_size, num_heads)   # per-head scalar score
        self.key       = nn.Linear(hidden_size, hidden_size)
        self.key_att   = nn.Linear(hidden_size, num_heads)
        self.transform = nn.Linear(hidden_size, hidden_size)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)   attn_mask: (B, 1, L) with 0 / -10000
        B, L, _ = x.shape
        Q = self.query(x)   # (B, L, D)
        K = self.key(x)     # (B, L, D)

        # ── step 1: pool Q → global query ──────────────────────────────
        q_scores = self.query_att(Q).transpose(1, 2) / self.head_size ** 0.5   # (B, H, L)
        if attn_mask is not None:
            q_scores = q_scores + attn_mask
        q_weight = torch.softmax(q_scores, dim=-1).unsqueeze(2)                 # (B, H, 1, L)
        q_heads  = self._to_heads(Q)                                            # (B, H, L, hd)
        q_bar    = torch.matmul(q_weight, q_heads)                              # (B, H, 1, hd)
        q_bar    = q_bar.transpose(1, 2).reshape(B, 1, -1).expand(-1, L, -1)   # (B, L, D)

        # ── step 2: interact q̄ with K ───────────────────────────────────
        R = K * q_bar                                                            # (B, L, D)

        # ── step 3: pool R → global key ────────────────────────────────
        k_scores = self.key_att(R).transpose(1, 2) / self.head_size ** 0.5     # (B, H, L)
        if attn_mask is not None:
            k_scores = k_scores + attn_mask
        k_weight = torch.softmax(k_scores, dim=-1).unsqueeze(2)                 # (B, H, 1, L)
        k_heads  = self._to_heads(R)                                            # (B, H, L, hd)
        k_bar    = torch.matmul(k_weight, k_heads)                              # (B, H, 1, hd)

        # ── step 4: k̄ * Q_heads + transform + residual to Q ───────────
        out = (k_bar * q_heads).transpose(1, 2).reshape(B, L, -1)              # (B, L, D)
        return self.transform(out) + Q                                           # residual


class FastAttention(nn.Module):
    """FastSelfAttention + BertSelfOutput (Linear → Dropout → LayerNorm + residual)."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn  = FastSelfAttention(hidden_size, num_heads)
        self.dense      = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        self_out = self.self_attn(x, attn_mask)
        return self.layer_norm(self.dropout(self.dense(self_out)) + x)


class FastformerLayer(nn.Module):
    """FastAttention + FFN (BertIntermediate + BertOutput style)."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.attention      = FastAttention(hidden_size, num_heads, dropout)
        self.intermediate   = nn.Linear(hidden_size, intermediate_size)
        self.output_dense   = nn.Linear(intermediate_size, hidden_size)
        self.output_norm    = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout        = nn.Dropout(dropout)
        self.act            = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attention(x, attn_mask)
        ffn_out  = self.output_dense(self.dropout(self.act(self.intermediate(attn_out))))
        return self.output_norm(ffn_out + attn_out)


class FastformerEncoder(nn.Module):
    """
    Position embeddings + N FastformerLayers + AttentionPooling.
    Matches their FastformerEncoder used inside UserEncoder.
    """

    def __init__(
        self,
        hidden_size:       int,
        num_heads:         int,
        num_layers:        int,
        intermediate_size: int,
        max_positions:     int,
        dropout:           float,
    ):
        super().__init__()
        self.position_emb = nn.Embedding(max_positions, hidden_size)
        self.layer_norm   = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout      = nn.Dropout(dropout)
        self.layers       = nn.ModuleList([
            FastformerLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.pooler = AttentionPooling(hidden_size, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)   mask: (B, L) bool (True=valid) or float (1=valid)
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x   = self.layer_norm(x + self.position_emb(pos))
        x   = self.dropout(x)

        # Build BERT-style extended mask: (B, 1, L) with 0 / -10000
        ext_mask = None
        float_mask = None
        if mask is not None:
            float_mask = mask.float() if mask.dtype == torch.bool else mask
            ext_mask   = (1.0 - float_mask).unsqueeze(1) * -10000.0  # (B, 1, L)

        for layer in self.layers:
            x = layer(x, ext_mask)

        return self.pooler(x, float_mask)


# ── encoders ───────────────────────────────────────────────────────────────────

class FastformerNewsEncoder(nn.Module):
    """
    Word embeddings → FastformerEncoder per view (title / abstract) →
    AttentionPooling over the two views.
    Mirrors their TextEncoder structure without the PLM backbone.
    """

    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        d          = cfg.model.word_emb_dim
        h          = cfg.model.num_heads * cfg.model.head_dim
        n_layers   = getattr(cfg.model, "num_fastformer_layers", 2)
        dropout    = cfg.model.dropout

        self.emb      = nn.Embedding(vocab_size, d, padding_idx=0)
        self.emb_proj = nn.Linear(d, h) if d != h else nn.Identity()
        self.dropout  = nn.Dropout(dropout)

        enc_kwargs = dict(
            hidden_size=h, num_heads=cfg.model.num_heads,
            num_layers=n_layers, intermediate_size=4 * h,
            max_positions=512, dropout=dropout,
        )
        self.title_enc    = FastformerEncoder(**enc_kwargs)
        self.abstract_enc = FastformerEncoder(**enc_kwargs)
        self.view_pooler  = AttentionPooling(h, dropout)

    def forward(self, titles: torch.Tensor, abstracts: torch.Tensor, **kwargs) -> torch.Tensor:
        t_mask = titles != 0
        a_mask = abstracts != 0

        t_emb = self.dropout(self.emb_proj(self.emb(titles)))
        a_emb = self.dropout(self.emb_proj(self.emb(abstracts)))

        t_vec = self.title_enc(t_emb, t_mask)                          # (B, h)
        a_vec = self.abstract_enc(a_emb, a_mask)                       # (B, h)

        views = torch.stack([t_vec, a_vec], dim=1)                     # (B, 2, h)
        return self.view_pooler(views)                                  # (B, h)


class FastformerUserEncoder(nn.Module):
    """
    FastformerEncoder over click history — matches their UserEncoder exactly.
    """

    def __init__(self, news_dim: int, cfg):
        super().__init__()
        n_layers = getattr(cfg.model, "num_fastformer_layers", 2)
        self.encoder = FastformerEncoder(
            hidden_size=news_dim,
            num_heads=cfg.model.num_heads,
            num_layers=n_layers,
            intermediate_size=4 * news_dim,
            max_positions=cfg.data.max_history + 1,
            dropout=cfg.model.dropout,
        )

    def forward(self, hist_vecs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(hist_vecs, history_mask)


# ── full model ─────────────────────────────────────────────────────────────────

class Fastformer(BaseRecommender):
    """
    UniUM-Fastformer: FastformerEncoder for both news and user encoding.
    news_dim = num_heads * head_dim — same contract as NRMS.
    """

    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.news_enc = FastformerNewsEncoder(vocab_size, cfg)
        self.user_enc = FastformerUserEncoder(cfg.model.num_heads * cfg.model.head_dim, cfg)

    def encode_news(self, titles, abstracts, categories, subcategories, **kwargs):
        return self.news_enc(titles, abstracts)

    def encode_user(self, hist_vecs, history_mask, **kwargs):
        return self.user_enc(hist_vecs, history_mask)
