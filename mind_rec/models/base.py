import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """Additive attention with a learned query vector (Bahdanau-style pooling)."""

    def __init__(self, in_dim: int, query_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D)  mask: (B, L) bool True=valid
        w = torch.tanh(self.proj(x)) @ self.query       # (B, L)
        if mask is not None:
            w = w.masked_fill(~mask, float("-inf"))
        w = torch.softmax(w, dim=-1)                    # (B, L)
        # rows where every position was masked produce NaN → zero them out
        w = torch.nan_to_num(w, nan=0.0)
        return (w.unsqueeze(-1) * x).sum(dim=1)         # (B, D)


class BaseRecommender(nn.Module):
    """
    Shared forward pass: encode history news → encode user → encode candidates → dot-product scores.
    Subclasses implement encode_news() and encode_user().
    """

    def encode_news(self, titles, abstracts, categories, subcategories, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def encode_user(self, hist_news_vecs, history_mask, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Returns logits (B, K) where K = 1 + neg_samples (train) or impression size (eval).
        """
        B, H = batch["history_titles"].shape[:2]
        K = batch["cand_titles"].shape[1]
        user_idx = batch.get("user_idx")

        hist_vecs = self.encode_news(
            batch["history_titles"].view(B * H, -1),
            batch["history_abstracts"].view(B * H, -1),
            batch["history_categories"].view(B * H),
            batch["history_subcategories"].view(B * H),
            user_idx=user_idx,
        ).view(B, H, -1)

        user_vecs = self.encode_user(hist_vecs, batch["history_mask"], user_idx=user_idx)

        cand_vecs = self.encode_news(
            batch["cand_titles"].view(B * K, -1),
            batch["cand_abstracts"].view(B * K, -1),
            batch["cand_categories"].view(B * K),
            batch["cand_subcategories"].view(B * K),
            user_idx=user_idx,
        ).view(B, K, -1)

        # (B, 1, D) x (B, D, K) → (B, 1, K) → (B, K)
        return (user_vecs.unsqueeze(1) @ cand_vecs.transpose(1, 2)).squeeze(1)
