# Oracle

A foundation model for soccer, built from scratch: an autoregressive, decoder-only
approach to predicting the next event in a match, trained on tokenized event data.

This repo is a **learning project**. The goal is two-fold — understand transformer /
LLM machinery by building the pieces by hand, and produce a native soccer event
model along the way. The current notebooks work up to that through a sequence of
MLP next-event predictors of increasing sophistication; the transformer is the
next step, not yet here.

## Approach

Each match is flattened into a sequence of events. Every event is a 7-field tuple

```
(is_home, event_type, start_x, start_y, time_elapsed, duration, result)
```

and the model predicts the next event one field at a time, autoregressively, so
each field can condition on the earlier fields of the same event (LEM-style
causal masking within the event, after Mendes-Neves et al., *A Scalable Approach
for Unified Large Events Models in Soccer*, ECML PKDD 2025). `x`/`y` are
discretized to one token per metre on a 105 × 68 pitch (SPADL's representation).

## Data

[StatsBomb open data](https://github.com/statsbomb/open-data) — **Premier League
2015/16**, 380 matches (`competition_id=2`, `season_id=27`). Loaded via
`statsbombpy` and `socceraction` (SPADL). `data/` is not committed; run notebooks
`00`–`03` to regenerate it.

## Run order

| # | Notebook | Output |
|---|---|---|
| 00 | `00_ingest.ipynb` | `data/statsbomb.csv`, `data/event_df.csv` (hits the network) |
| 01 | `01_build_stage.ipynb` | `data/stage.csv` — SPADL actions + running match state |
| 02 | `02_stage_to_df.ipynb` | `data/masked_df.csv` — each event → 7 causally-masked rows |
| 03 | `03_splits.ipynb` | `train_set.csv` / `val_set.csv` / `test_set.csv` (80/10/10 by game) |
| 04 | `04_eda.ipynb` | pitch maps, transition matrices, timing, possession flow |
| 05 | `05_mlp_v1_shared_vocab.ipynb` | v1 model + baseline checkpoints |
| 06 | `06_mlp_v1_evaluation.ipynb` | human baseline quiz + v1 analysis |
| 07 | `07_mlp_v2_deeper.ipynb` | v2 (deeper) model |
| 08 | `08_mlp_v3_handwritten_backprop.ipynb` | v3 model (hand-written backward pass) |
| 09 | `09_mlp_v4_embeddings_shared_head.ipynb` | v4 model (categorical embeddings) |
| 10 | `10_mlp_v5_embeddings_disjoint_heads.ipynb` | v5 model — **current best** |
| 11 | `11_scaling_laws.ipynb` | data / parameter scaling sweeps |
| 12 | `12_benchmarking.ipynb` | full-match simulation vs. real score distribution |
| 13 | `13_naive_baseline.ipynb` | mode / median / marginal-CE floor for every model |
| 14 | `14_full_event_evaluation.ipynb` | autoregressive vs teacher-forced, baseline vs v5 |

The model notebooks (05, 07–10) each read the split CSVs and are independent of
one another; 06 depends on 05's checkpoints, and 11–12 use v5's architecture /
checkpoint. `13`–`14` are evaluation add-ons (numbered last only because they
were added last).

## The model progression

| | change from previous | how it's built |
|---|---|---|
| **v1** | 2-layer MLP (`37→80→107`), all 7 fields sharing one 107-wide output vocab, categorical fields as raw scalar codes, `BCEWithLogits` | `torch` autograd |
| **v2** | deeper (`37→96→96→96→107`), everything else held fixed | `torch` autograd |
| **v3** | same net as v2; 500k steps with a stepped lr schedule (v2: 1M steps at a fixed lr) | **backward pass by hand**, no autograd |
| **v4** | `is_home`/`event`/`result` (+ lags) via embedding tables instead of scalar codes | back to autograd |
| **v5** | **7 disjoint per-field output heads** (one softmax + cross-entropy each) instead of a shared vocab; continuous fields normalized | autograd |

Each step was made in isolation so any change in results is attributable to that
one change.

## Results

Per-field metrics on the validation split, each field predicted given the true
values of the earlier fields (accuracy for `is_home`/`event`/`result`, mean
absolute error otherwise):

| field | metric | naive | v1 | v2 | v3 | v4 | **v5** |
|---|---|---|---|---|---|---|---|
| is_home | acc | 51.8% | 82.3% | 94.2% | 93.9% | 94.5% | **95.0%** |
| event | acc | 43.0% | 66.1% | 69.8% | 71.0% | 72.4% | **75.5%** |
| result | acc | 82.7% | 82.7% | 86.6% | 85.6% | 86.7% | **88.4%** |
| start_x | MAE | 20.6 | 12.26 | 8.62 | 9.00 | 7.98 | **6.49** |
| start_y | MAE | 17.5 | 10.43 | 8.82 | 9.06 | 8.06 | **6.90** |
| time_elapsed | MAE | 2.19 | 1.99 | 1.67 | 1.74 | 1.60 | **0.91** |
| duration | MAE | 0.90 | 0.82 | 0.87 | 0.81 | 0.79 | **0.75** |

("naive" = always predict the training mode / median — see `13_naive_baseline.ipynb`.
The numbers above are teacher-forced, on val, from a full re-run of every
notebook. `14_full_event_evaluation.ipynb` lays out both models × both modes.
v4 and v5 reproduce to within 0.01 across runs; v1 and v3 drift by up to
~0.35 MAE / ~1 point accuracy, so treat the v1–v3 gaps as approximate. v3 is
v2's net with a hand-written backward pass, so it isn't expected to beat it.)

### Test set (held out, evaluated once)

v5 vs the naive baseline on the 38-game test split — the model's dev never
touched it. Two models × two modes: **AR** = autoregressive (the model feeds
itself its own predictions down the chain), **TF** = teacher-forced (true
earlier fields given). Bold = better model within a mode.

| field | metric | naive · AR | v5 · AR | naive · TF | v5 · TF |
|---|---|---|---|---|---|
| is_home | acc | 0.516 | **0.949** | 0.516 | **0.949** |
| event | acc | 0.433 | **0.734** | 0.433 | **0.753** |
| start_x | MAE | 20.796 | **6.731** | 20.796 | **6.476** |
| start_y | MAE | 17.574 | **7.410** | 17.574 | **6.870** |
| time_elapsed | MAE | 2.233 | **1.434** | 2.233 | **0.930** |
| duration | MAE | 0.900 | **0.887** | 0.900 | **0.753** |
| result | acc | **0.824** | 0.821 | 0.824 | **0.881** |

The two `naive` columns are identical (mode/median ignores context). Test
tracks val to within ~0.003 accuracy / ~0.03 MAE on every field — no
overfitting to the validation split. v5 beats the baseline on 7/7 fields
teacher-forced and 6/7 autoregressive (losing only `result`, 0.821 vs the
"always success" floor of 0.824). Summed cross-entropy on test: 9.575
teacher-forced, 11.249 autoregressive (training-marginal baseline: 14.96).

### Scaling (`11_scaling_laws.ipynb`)

v5 architecture, 300k steps per run, final val loss (summed cross-entropy):

| train data | val loss | | hidden layers | params | val loss |
|---|---|---|---|---|---|
| 25% | 10.072 | | `[80]` | 33.7k | 11.051 |
| 50% | 10.042 | | `[96, 96]` | 49.7k | 10.444 |
| 100% | 10.015 | | `[96, 96, 96]` | 59.0k | 10.015 |

Loss falls with both, but parameters matter far more at this scale: 4× the
data gains 0.06 nats, going from one hidden layer to three gains 1.04.

Headline findings from the evaluation notebooks:

- **The shared 107-wide vocabulary was the main bottleneck.** With every field
  sharing an output space, class value `1` simultaneously meant `home`, `pass`,
  `time_elapsed=1` and `duration=1`, so frequency from one field bled into
  others — producing structurally impossible predictions (e.g. `foul → success`
  ~53% of the time under v3). Splitting into disjoint per-field heads (v5) fixed
  this and improved every field at once — more than embeddings or normalization
  did individually.
- **Two kinds of failure, needing different fixes.** Classes with real support
  (hundreds of examples) became genuinely learnable once the representation was
  fixed; classes with only tens of examples (`shot_penalty`, `owngoal`,
  `red_card`) stay unlearnable regardless of architecture — that bucket needs
  more matches.
- After the representation fixes, **structural predictability matters more than
  raw example count**: near-deterministic events given the 3-event lookback
  (set pieces, period boundaries) are learned well at any support size, while
  open-play actions with several plausible continuations (`take_on`, `bad_touch`,
  `cross`) stay hard even with 1,000+ examples.

## Status / next

- Move from MLP to the decoder-only transformer.
- Add team / player conditioning (the model currently has no identity in its
  inputs, so it can only be benchmarked at the population level).
- `12_benchmarking.ipynb`'s full-match simulation never generates `full_end`:
  the match clock runs past 90 minutes without closing the half, so
  simulations hit the 8,000-event cap and the score-distribution comparison
  isn't meaningful yet.

## Environment

Python 3.12 venv (`uv venv --python 3.12`), then
`uv pip install -r requirements.txt`.
