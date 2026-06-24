# MIND News Recommendation — Training Documentation

## 1. Dataset: MINDsmall

| Split | File | Contents |
|---|---|---|
| Train | `MINDsmall_train/MINDsmall_train/` | `news.tsv`, `behaviors.tsv`, `entity_embedding.vec` |
| Dev / Test | `MINDsmall_dev/MINDsmall_dev/` | same structure |

**news.tsv** columns: `news_id`, `category`, `subcategory`, `title`, `abstract`, `url`, `title_entities`, `abstract_entities`

**behaviors.tsv** columns: `impression_id`, `user_id`, `time`, `history` (space-separated clicked news IDs), `impressions` (space-separated `newsID-click_label` tokens)

---

## 2. Preprocessing Pipeline

```
news.tsv + behaviors.tsv
        │
        ▼
parse_news / parse_behaviors          (data/dataset.py)
        │
        ▼
Build index maps:
  cat2idx    – category  → int (1-indexed, 0 = padding)
  subcat2idx – sub-cat   → int
  user2idx   – user_id   → int
        │
        ▼
Vocab.build(title + abstract, min_freq=1)   (data/vocab.py)
  Word → index; 0 = <PAD>, 1 = <UNK>
  word_emb_dim = 300  (optionally initialised from GloVe)
        │
        ▼
MINDTrainDataset   – one sample = (1 pos + K neg) per impression
MINDEvalDataset    – one sample = all candidates in impression
```

**Sequence truncation defaults** (from `DataConfig`):

| Field | Max length |
|---|---|
| Title | 20 tokens |
| Abstract | 50 tokens |
| Click history | 50 articles |

---

## 3. Training Loop (`trainer.py` / `sweep.py`)

```
for each epoch:
    for each batch:
        scores = model(batch)          # (B, 1 + K)   logits
        loss   = CrossEntropyLoss(scores, label=0)   # label 0 = positive always first
        loss.backward()
        clip_grad_norm(max=1.0)
        Adam.step()

    val_loss = CrossEntropyLoss on held-out val split
    if val_loss < best:  save checkpoint
    else:                patience_counter++
    if patience_counter >= PATIENCE: early stop
```

**Optimiser:** Adam, no weight decay  
**Loss:** Cross-entropy over `(1 pos, K neg)` candidates per impression  
**Gradient clipping:** max norm = 1.0  
**Early stopping patience:** 3 epochs (sweep) / none (single-run `main.py`)  
**AMP (mixed precision):** enabled automatically on CUDA  

### Sweep-specific additions

The sweep (`sweep.py`) splits training behaviors **90 / 10** (train / val) before building datasets, so the held-out dev set is never touched during HP selection:

```
behaviors.tsv (train)
    ├── 90 % → MINDTrainDataset   (gradient updates)
    └── 10 % → val_train_ds      (val_loss for early stopping)
                val_eval_ds       (AUC/MRR/nDCG for best-checkpoint selection)

behaviors.tsv (dev)  → test_ds   (final reporting only)
```

---

## 4. Model Architectures

All models share the same two-tower design:

```
[News Encoder]  →  news_vec  (D-dim)
[User Encoder]  →  user_vec  (D-dim)
score = dot(user_vec, cand_news_vec)
```

### 4.1 NRMS — Neural News Recommendation with Multi-Head Self-Attention

> Wu et al., ACL-EMNLP 2019

```
News Encoder
  title tokens  →  Embedding(300)  →  MHSA  →  AdditiveAttention  →  title_vec (400)
  abstract tokens → Embedding(300) →  MHSA  →  AdditiveAttention  →  abs_vec   (400)
  [title_vec, abs_vec]             →  AdditiveAttention            →  news_vec  (400)

User Encoder
  [news_vec₁ … news_vecH]         →  MHSA  →  AdditiveAttention  →  user_vec  (400)
```

`news_dim = num_heads × head_dim = 20 × 20 = 400`

Shared word embedding between title and abstract encoders.  
**Params:** ~19.8 M

---

### 4.2 NAML — Neural News Recommendation with Attentive Multi-View Learning

> Wu et al., IJCAI 2019

```
News Encoder
  title tokens    →  Embedding(300) →  CNN(filters=400, k=3) →  AdditiveAttention  →  title_vec   (400)
  abstract tokens →  Embedding(300) →  CNN(filters=400, k=3) →  AdditiveAttention  →  abs_vec     (400)
  category        →  Embedding(400)                                                 →  cat_vec     (400)
  sub-category    →  Embedding(400)                                                 →  subcat_vec  (400)
  [title, abstract, cat, subcat]   →  AdditiveAttention                            →  news_vec    (400)

User Encoder
  [news_vec₁ … news_vecH]         →  AdditiveAttention                            →  user_vec    (400)
```

Shared word embedding between title and abstract CNNs.  
**Params:** ~19.4 M

---

### 4.3 LSTUR — Long- and Short-Term User Representation

> An et al., ACL 2019

```
News Encoder
  title tokens  →  Embedding(300)  →  CNN(filters=400)  →  AdditiveAttention  →  news_vec (400)

User Encoder  (two fusion modes)
  user_id → UserEmbedding(50)  →  long-term user rep

  mode = "ini":
    long_term  →  Linear(50→400)  →  GRU initial hidden state
    hist_vecs  →  GRU(400)       →  last valid hidden  →  user_vec (400)

  mode = "con":
    hist_vecs  →  GRU(400)       →  short_term (400)
    [short_term ‖ long_term]     →  Linear(450→400)   →  user_vec (400)
```

**Params:** ~24.4 M (ini) / ~24.5 M (con)

---

### 4.4 NPA — Neural News Recommendation with Personalized Attention

> Wu et al., KDD 2019

```
News Encoder  (user-conditioned)
  user_id → UserEmbedding(50) → Linear(50→200)  →  user_query (200)
  title   → Embedding(300)   → CNN(filters=400) →  word_feats (L × 400)
  attn_score = word_proj(word_feats) · user_query          (personalised)
  news_vec = weighted_sum(word_feats, attn_score)          →  (400)

User Encoder  (user-conditioned)
  user_id → UserEmbedding(50) → Linear(50→200)  →  user_query (200)
  hist_vecs (H × 400)  →  news_proj(400→200) · user_query  →  attn scores
  user_vec = weighted_sum(hist_vecs, attn_scores)           →  (400)
```

Both the word-level and news-level attention are personalised by the same user embedding.  
**Params:** ~28.2 M

---

### 4.5 Fastformer — Additive Attention Transformer

```
News Encoder
  title tokens  →  Embedding(300)  →  N × FastformerLayer  →  AdditiveAttention  →  news_vec (400)

FastformerLayer:
  Q, K, V = linear projections
  global_q = weighted_mean(Q)           (additive attention over Q)
  p = global_q ⊙ K                      (element-wise, then global_k = weighted_mean)
  u = global_k ⊙ V
  output = linear(u) + residual

User Encoder
  [news_vec₁ … news_vecH]  →  AdditiveAttention  →  user_vec (400)
```

---

### Shared Component: AdditiveAttention

Used by all models to pool a sequence into a single vector:

```
score_i = v · tanh(W · h_i + b)      (v, W, b are learned)
α = softmax(score, mask)
output = Σ αᵢ hᵢ
```

---

## 5. Hyperparameter Sweep

### Sweep grid

| Model | Configs | Key HPs varied |
|---|---|---|
| NRMS | 8 | lr ∈ {1e-4, 3e-4, 1e-3}, dropout ∈ {0.1, 0.2, 0.3}, batch ∈ {64, 128}, heads: 20×20 vs 16×25 |
| NAML | 4 | lr ∈ {1e-4, 3e-4}, dropout ∈ {0.2, 0.3}, kernel ∈ {3, 5} |
| LSTUR | 4 | lr ∈ {1e-4, 3e-4}, mode ∈ {ini, con} |
| NPA | 4 | lr ∈ {1e-4, 3e-4}, dropout ∈ {0.1, 0.2, 0.3}, batch ∈ {64, 128} |

Fixed across all: `num_filters = 400`, `word_emb_dim = 300`, `max_history = 50`, `neg_samples = 4`, `seed = 42`

### Outputs

```
results/sweep/
  sweep_summary_<ts>.csv    – one row per run: HPs, best_epoch, val metrics, test metrics, checkpoint path
  epoch_metrics_<ts>.csv    – one row per (run, epoch): train_loss, val_loss, val_auc, …
  checkpoints/
    <model>_run<id>_best.pt – state_dict at best val_loss epoch
```

---

## 6. Evaluation Metrics

Computed per-impression, then averaged:

| Metric | Formula |
|---|---|
| **AUC** | sklearn `roc_auc_score` |
| **MRR** | 1 / rank of first relevant item |
| **nDCG@5** | normalised DCG with discount log₂(rank+1), top-5 |
| **nDCG@10** | same, top-10 |

Impressions with all-positive or all-negative labels are excluded.

---

## 7. Sweep Results (Best per Model)

| Run | Model | lr | dropout | batch | HPs | Best epoch | Val AUC | Test AUC | Test MRR | Test nDCG@10 | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | **NRMS** | 1e-4 | 0.2 | 64 | heads=16×25 | — | 0.7283 | **0.6279** | 0.3352 | 0.3817 | 19.8 M |
| 18 | NRMS | 1e-4 | 0.1 | 128 | heads=20×20 | 9 | 0.7234 | 0.6248 | 0.3370 | 0.3813 | 19.8 M |
| 13 | NRMS | 1e-4 | 0.2 | 64 | heads=20×20 | 19 | 0.7289 | 0.6244 | 0.3317 | 0.3779 | 19.8 M |
| 5 | LSTUR | 1e-4 | 0.2 | 64 | mode=ini | 8 | 0.7104 | 0.5972 | 0.3116 | 0.3541 | 24.4 M |
| 10 | NPA | 3e-4 | 0.2 | 64 | — | 6 | 0.7062 | 0.5966 | 0.3053 | 0.3530 | 28.2 M |
| 3 | NAML | 1e-4 | 0.3 | 64 | k=3 | 5 | 0.6717 | 0.5790 | 0.2946 | 0.3409 | 19.4 M |

**Run 20 (NRMS, 16 heads × 25 dim)** is the best overall and is used as the scorer in the diversity re-ranking notebook.

---

## 8. Quick-Start Commands

```bash
# Single model run
cd mind_rec/
python main.py --model nrms --epochs 10 --lr 1e-4 --device mps

# Full HP sweep (all 4 models, ~20 runs)
python sweep.py --device mps --skip_epoch_auc

# Smoke test (1 config, 1 epoch, 5 steps)
python sweep.py --smoke_test --device cpu

# Subset models
python sweep.py --models nrms naml --device mps
```

---

## 9. File Map

```
mind_rec/
├── main.py              – single-model CLI entry point
├── sweep.py             – full HP sweep with early stopping + CSV logging
├── trainer.py           – train() and evaluate() used by main.py
├── evaluate.py          – AUC, MRR, nDCG metric computation
├── config.py            – DataConfig, ModelConfig, TrainConfig dataclasses
├── data/
│   ├── dataset.py       – parse_news, parse_behaviors, MINDTrainDataset, MINDEvalDataset
│   └── vocab.py         – Vocab (word→idx, GloVe loading)
├── models/
│   ├── base.py          – AdditiveAttention, BaseRecommender
│   ├── nrms.py          – NRMS
│   ├── naml.py          – NAML
│   ├── lstur.py         – LSTUR
│   ├── npa.py           – NPA
│   └── fastformer.py    – Fastformer
└── results/sweep/
    ├── sweep_summary.csv
    ├── epoch_metrics.csv
    └── checkpoints/     – best .pt per run
```
