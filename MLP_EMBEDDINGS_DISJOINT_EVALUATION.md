# mlp_embeddings_disjoint Evaluation Summary

Results for the disjoint-per-field-head embeddings MLP
(`mlp_embeddings_disjoint_embeddings.ipynb`), evaluated against the val
split and compared to the two prior models: `mlp_hand.ipynb` (shared
107-wide scalar-input vocabulary) and the intermediate `mlp_embeddings`
shared-output-vocabulary version (categorical embeddings, but still one
shared `BCEWithLogits` head). This is the current best model and, absent a
move to a transformer architecture, likely the final MLP iteration.

## Setup

- **Architecture**: same 4-layer shared trunk as prior MLPs (96 hidden,
  Kaiming-scaled hidden weights, zeroed biases) feeding **7 separate
  per-field output heads** instead of one shared 108-wide vocabulary — one
  softmax + cross-entropy per field (`is_home`:3, `event`:24, `start_x`:107,
  `start_y`:70, `time_elapsed`:106, `duration`:37, `result`:7 classes),
  routed per-row via the `step` column (which field is currently masked).
  Total ~59K parameters, dominated by the wide-vocab heads (`start_x`/
  `time_elapsed` alone are ~10K params each).
- **Input**: `is_home`/`event`/`result` embedded (`emb_size=3`, shared table
  across current + 3 lag positions per field); `start_x`/`start_y`/
  `time_elapsed`/`duration`/`minute`/`second` standardized to
  `mean=10, std=1` (stats from train only, excluding `-1`/`-2` sentinels,
  which are left untouched — real data shifted safely away from them rather
  than sentinels remapped to make room). `period_id` and goal/card counts
  left raw (already small).
- **Training**: 1,000,000 steps, batch size 1,000, lr schedule
  `0.001 → 0.0001 → 0.00001` (dropped ~100x from the original `0.1` after a
  real gradient explosion — see incident note below).
- **Loss function change**: `F.cross_entropy` per field (proper single-label
  categorical loss, own softmax per field) instead of one shared
  `BCEWithLogits` treating 108 positions as independent binary decisions.
  Not directly comparable in scale to the older single-loss numbers — sums
  7 independent `ln(num_classes)`-scale losses instead of one shared,
  averaged one.

## Incident: gradient explosion, and the normalization fix

Early in training with the original `lr=0.1` schedule (inherited from prior
models), loss exploded from `23` to `7.7e21` within 5 steps — confirmed via
direct trace to be a real numerical overflow (`inf` gradient norm at step
4), not a benign recovery. Root cause: raw, unnormalized continuous inputs
(`minute` up to 101, `duration` up to 68) combined with the new loss's
larger scale (sum of 7 cross-entropies vs. one shared BCE) produced large
gradients from step 0, and plain SGD with no clipping compounded them every
step until the numbers overflowed float32. Fixed by (a) dropping `lr` by
~100x and (b) the continuous-field normalization described above — both
needed, since normalization alone doesn't cap gradient *steps*, only input
*magnitude*.

## Incident: checkpoint file collision

`mlp_embeddings_shared_output.ipynb` (a from-scratch reconstruction of the
pre-disjoint-heads checkpoint, built after the original was accidentally
overwritten by unrelated debugging) and this notebook both originally
targeted `data/mlp_embeddings_checkpoint.pt`. That path was overwritten
multiple times across the two notebooks, including once by an unrelated
smoke test. This model's checkpoint now lives at its own path,
`data/mlp_embeddings_disjoint_checkpoint.pt`, saving `output_heads` as a
dict (not a stray single `W4`/`b4`, which was the original bug) and using
self-documenting embedding-table keys (`C_event`/`C_is_home`/`C_result`,
not `C1`/`C2`/`C3`) — lesson: give every architecture variant its own
checkpoint filename from the start.

## Headline: per-field accuracy / MAE (val), all three models

| field | metric | mlp_hand | mlp_embeddings (shared) | **mlp_embeddings_disjoint** |
|---|---|---|---|---|
| is_home | accuracy | 94.1% | 94.5% | **95.05%** |
| event | accuracy | 71.3% | 72.4% | **75.51%** |
| result | accuracy | 86.4% | 86.7% | **88.44%** |
| start_x | MAE | 8.81 | 7.98 | **6.50** |
| start_y | MAE | 8.71 | 8.06 | **6.89** |
| time_elapsed | MAE | 1.68 | 1.60 | **0.91** |
| duration | MAE | 0.80 | 0.79 | **0.75** |

Every field improved over both prior models, several substantially
(`time_elapsed` MAE nearly halved; `event` accuracy up ~3pp; `start_x`/
`start_y` MAE down 15-18%). Val loss (sum of 7 field cross-entropies):
`9.573`, close to the training-subset loss (`9.510`) — small, healthy
train/val gap.

## `event` deep dive

Full precision/recall/F1, all 24 classes, sorted by support:

| event | support | predicted | precision | recall | f1 |
|---|---|---|---|---|---|
| pass | 32212 | 33299 | 0.787 | 0.813 | 0.800 |
| dribble | 27957 | 31701 | 0.792 | 0.898 | 0.842 |
| clearance | 2231 | 1808 | 0.493 | 0.399 | 0.441 |
| throw_in | 1768 | 1436 | 0.523 | 0.425 | 0.469 |
| tackle | 1524 | 1709 | 0.721 | 0.809 | 0.763 |
| take_on | 1314 | 13 | 0.308 | 0.003 | 0.006 |
| interception | 1313 | 1224 | 0.423 | 0.395 | 0.408 |
| bad_touch | 1149 | 42 | 0.381 | 0.014 | 0.027 |
| shot | 1021 | 480 | 0.500 | 0.235 | 0.320 |
| foul | 906 | 566 | 0.475 | 0.297 | 0.365 |
| cross | 901 | 159 | 0.396 | 0.070 | 0.119 |
| goalkick | 641 | 723 | 0.506 | 0.571 | 0.537 |
| freekick_crossed | 454 | 487 | 0.583 | 0.626 | 0.604 |
| freekick_short | 420 | 484 | 0.537 | 0.619 | 0.575 |
| corner_crossed | 340 | 360 | 0.411 | 0.435 | 0.423 |
| keeper_save | 229 | 183 | 0.361 | 0.288 | 0.320 |
| keeper_claim | 168 | 1 | 0.000 | 0.000 | 0.000 |
| corner_short | 71 | 0 | 0.000 | 0.000 | 0.000 |
| keeper_punch | 54 | 0 | 0.000 | 0.000 | 0.000 |
| half_end | 38 | 32 | 0.500 | 0.421 | 0.457 |
| full_end | 38 | 44 | 0.500 | 0.579 | 0.537 |
| kick_off | 38 | 38 | 1.000 | 1.000 | 1.000 |
| shot_freekick | 33 | 42 | 0.429 | 0.545 | 0.480 |
| shot_penalty | 11 | 0 | 0.000 | 0.000 | 0.000 |

Macro precision **0.443** / macro recall **0.393** (up from `mlp_hand`'s
0.288 / 0.188 — recall more than doubled) / micro accuracy **0.755**.
Classes never predicted: 3/24 (down from 6/24).

**Case study — mean predicted probability on the true class**, same
deterministic/rare classes tracked since `mlp_hand`:

| class | mlp_hand | disjoint |
|---|---|---|
| kick_off | 96.3% | **99.4%** (now literally perfect, F1=1.000) |
| half_end | 9.3% | **48.9%** (5x) |
| full_end | 12.4% | **51.6%** (4x) |
| shot_freekick | 0.6% | **35.0%** (60x) |
| shot_penalty | 0.05% | **14.8%** (300x, still low in absolute terms) |

**Outlier pattern — support size no longer predicts performance.**
`shot_freekick` (n=33, F1=0.480) beats `clearance`/`interception`/
`throw_in` (n=1300-2200, F1=0.41-0.47) on 1/40th to 1/70th the data.
`kick_off`/`half_end`/`full_end`/`freekick_crossed`/`freekick_short` (all
tiny-to-mid support) are the best-performing classes outside `pass`/
`dribble`/`tackle`. Meanwhile `take_on` (n=1314, F1=0.006) and `bad_touch`
(n=1149, F1=0.027) remain essentially unlearned despite substantial
support — more data than several classes that improved a lot. The dividing
line looks structural: set-piece/period-boundary events (near-deterministic
given the 3-event lookback) do well regardless of support; open-play
actions with several plausible continuations (`take_on`, `bad_touch`,
`cross`) stay hard regardless of support. Support predicts performance only
within these two groups, not across them.

## `result` deep dive

| result | support | predicted | precision | recall | f1 |
|---|---|---|---|---|---|
| success | 61859 | 66100 | 0.903 | 0.965 | 0.933 |
| fail | 12627 | 8617 | 0.737 | 0.503 | 0.598 |
| offside | 135 | 0 | 0.000 | 0.000 | 0.000 |
| boundary | 114 | 114 | 1.000 | 1.000 | 1.000 |
| yellow_card | 93 | 0 | 0.000 | 0.000 | 0.000 |
| owngoal | 3 | 0 | 0.000 | 0.000 | 0.000 |

Macro precision 0.377 / macro recall 0.353 / micro accuracy 0.884.

**The "impossible outcome" bug is essentially fixed.** `foul` predicted as
`success` (structurally impossible — a foul is never a successful action):
`mlp_hand` 52.8%, shared-output `mlp_embeddings` 82.0% (worse), **this
model 0.7%** — now correctly predicts `fail` 99.3% of the time. Strong
confirmation that this was a shared-vocabulary artifact, not a data or
capacity problem.

**Cards show a real, if partial, qualitative shift.** All 93 true
`yellow_card` rows are misclassified (0% recall, same as prior models) —
but 92 of 93 are now called `fail` (a sensible near-miss — a card is a kind
of failure) versus being called `success` before. `offside` doesn't show
this same shift (still leans `success` over `fail`). Same two-bucket
pattern as `event`: classes with real support (`success`/`fail`/
`boundary`) are now well-separated; the truly rare ones (<135 examples)
still lack enough signal to win their own argmax, disjoint heads or not.

## `time_elapsed` deep dive

MAE improvement (1.68 → 0.91) traces to fixing the same collapse pattern
found in `event`/`result`, just showing up as a numeric-field symptom:

| | mlp_hand | disjoint |
|---|---|---|
| unique values predicted | 14/101 (14%) | **47/106 (44%)** |
| predicted std | 1.24 | **5.09** (true std: 6.68) |

More than 3x the class diversity; predicted spread now closes most of the
gap to the true spread, instead of collapsing to "always guess ~1."

**Performance degrades cleanly with true-value rarity, not uniformly.**
For the ~90% of rows where the true value is 0-3, MAE is 0.16-0.48
(near-perfect). For rarer, larger true values, error grows steadily and
the model systematically under-predicts (mean residual `-0.405` overall;
at true=18, mean prediction is 13.51) — classic regression-to-the-mean:
hedging toward common short values rather than committing to a rare long
one.

**Attractor spike at `time_elapsed=30`.** Predicted 398 times in val vs.
a true rate of only 81 (~5x over-representation) — and `30` isn't
naturally more common than its neighbors in the true data (25-35 all
occur at similar rates, 525-835 in train). The model has latched onto this
one specific tail value as a fallback for some cluster of contexts, at the
expense of its equally-common neighbors — not yet traced to what those
contexts have in common.

## Key takeaways

- Disjoint per-field output heads (removing the shared 108-wide vocabulary)
  was the single highest-leverage architectural fix identified across this
  whole project — it improved every field, fixed the specific
  cross-field-bleeding bugs documented for `mlp_hand` (`foul`→`success`,
  numeric-field collapse), and did so more than embeddings or normalization
  did individually.
- The same two-bucket failure taxonomy from the original root-cause
  synthesis still holds, just with the boundary shifted: classes/values
  with real support (hundreds of examples) are now genuinely learnable
  regardless of architecture; classes with only tens of examples
  (`shot_penalty`, `owngoal`, `red_card`) remain unlearnable no matter what
  representation fix is applied — that bucket needs more data, not a
  better architecture.
- Support size stopped being the dominant predictor of per-class
  performance — structural predictability from the 3-event lookback
  (set pieces, period boundaries) now matters more than raw example count.
  This is a new, useful diagnostic lens for any future work: a class doing
  badly despite ample data is a sign of genuine ambiguity in the task, not
  a data problem.
