# mlp_hand Evaluation Summary

Results for the hand-implemented next-event-prediction MLP (`mlp_hand.ipynb`),
evaluated against a zeroed-input baseline. Same task and data pipeline as
`mlp.ipynb` (see `MLP_EVALUATION.md`), but a deeper architecture, trained for
longer.

## Setup

- **Architecture**: 4-layer MLP, 37 input features → 96 → 96 → 96 (ReLU) →
  107 output logits, trained with `binary_cross_entropy_with_logits` over the
  same shared 107-wide vocabulary as `mlp.ipynb` — all 7 event fields
  (`is_home`, `event`, `start_x`, `start_y`, `time_elapsed`, `duration`,
  `result`) share one output/class-index space.
- **Training**: 1,000,000 steps, batch size 512, seed 42. LR schedule:
  `0.1` (0–500K) → `0.01` (500K–850K) → `0.001` (850K–1M). Kaiming-scaled
  hidden-layer weights, zeroed biases, tiny (`*0.001`) final-layer weight
  init.
- **Zeroed-input baseline**: same 4-layer architecture, separate parameters,
  trained 500,000 steps with every input feature forced to zero. Since
  gradients w.r.t. earlier layers are then always exactly zero (input and all
  intermediate biases are zero, so every hidden activation is pinned at
  zero), this model reduces mathematically to a single learned bias vector —
  the floor for "did the real model learn anything from the input at all."
- **Checkpoints**: `data/mlp_hand_checkpoint.pt` (real model),
  `data/mlp_hand_baseline_checkpoint.pt` (baseline).

## Loss

| | val loss |
|---|---|
| zeroed-input baseline | 0.0360 |
| **trained model** | **0.0206** |

~43% lower than the floor — learning from input, not just memorizing target
frequencies. Baseline converges (as `mlp.ipynb`'s did) to
`is_home=home, event=pass, start_x=1, start_y=1, time_elapsed=1, duration=1,
result=fail` — the shared-vocabulary artifact: class value `1` is
simultaneously the code for `home`/`pass`/`time_elapsed=1`/`duration=1`, so a
bias-only fit gravitates toward it almost everywhere.

## Per-field accuracy / MAE (given true context, val)

| field | metric | value |
|---|---|---|
| is_home | accuracy | 94.1% |
| event | accuracy | 71.3% |
| result | accuracy | 86.4% |
| start_x | MAE | 8.81 |
| start_y | MAE | 8.71 |
| time_elapsed | MAE | 1.68 |
| duration | MAE | 0.80 |

## Categorical collapse / distribution match

Checked whether the model uses the full class space per field, and how
closely its predicted class distribution matches the true one (val,
restricted to each field's valid classes):

| field | classes predicted | classes present | TV distance (pred vs true) |
|---|---|---|---|
| is_home | 3/3 | 3/3 | 0.0036 (near-perfect match) |
| event | **18/24** | 24/24 | 0.1235 |
| result | 6/7 | 7/7 | 0.0821 |

Real improvement over `mlp.ipynb`'s 2-layer model, which had collapsed
`event` to essentially 2 classes (`pass`/`dribble`, 99.3% of predictions) and
`result` to a near-constant "always success" classifier. Not fully fixed
though — see confusion matrix / precision-recall below for what "18/24
classes used" is hiding.

## `event` confusion matrix + precision/recall (val)

Macro precision/recall across all 24 event classes: **0.288 / 0.188** —
vs. micro (= overall accuracy) **0.713**. The gap is the headline: accuracy
looks solid because it's dominated by `pass`/`dribble` (support ~30K each,
F1 0.77–0.81); averaged equally across all 24 classes the model does badly.
**9 of 24 classes are never predicted at all** (`take_on`, `bad_touch`,
`goalkick`, `corner_crossed`, `keeper_save`, `keeper_claim`,
`corner_short`, `keeper_punch`, `shot_freekick`, `shot_penalty` — precision
and recall both 0).

Confusion isn't random — most rare classes get absorbed into `pass`/
`dribble`, but goalkeeper/clearance-adjacent actions (`goalkick`,
`keeper_save`, `keeper_claim`, `keeper_punch`) specifically collapse into
`clearance` instead — a semantically sensible but under-differentiated
grouping (the model has learned "ball dealt with defensively" as one
category, just can't split it further).

## Case study: deterministic vs. genuinely-unlearned rare classes

Compared mean predicted probability on the *true* class (not just argmax) to
separate two different failure modes hiding under "0% recall":

| event | true class mean prob | top-1 confidence | what happens instead |
|---|---|---|---|
| kick_off | 96.3% | — | learned perfectly — unique signature (`is_home=-2`, full history `-2`, `minute=0`) |
| half_end | 9.3% | 81.6% | confidently misassigned to `kick_off` (shares `is_home=-2`, but `minute`/`period_id` — the real separator — isn't being used) |
| full_end | 12.4% | 78.6% | same as half_end |
| shot_freekick | 0.6% | 39.2% | genuinely unlearned — diffuse, low-confidence guessing among `dribble`/`freekick_crossed`/`throw_in` |
| shot_penalty | 0.05% | 34.5% | same — no rule found at all |

`kick_off`/`half_end`/`full_end` all have ~304 clean training examples
each — not data-starved — yet half/full-end still collapse into kick_off.
`shot_penalty`/`shot_freekick` (73/323 train examples) additionally need a
conjunction of `event-1==foul` AND `is_home-1` AND `x-1`/`y-1` (pitch
coordinates aren't team-relative, so "near an end" only means something
combined with which team was fouled) — a harder pattern that too few
examples on raw unnormalized scalars never surfaced.

## `result` predictions grouped by `event` (val, step==6 — `event` is a
revealed true input feature at this point)

Full per-event breakdown lives in conversation history / can be regenerated
via `field_confusion_matrix`-style code in the notebook. Headline findings:

- **Structurally impossible predictions**: `foul` → `success` predicted
  52.8% of the time when **true rate is 0%** (a foul, by definition, is
  never a "successful" action; true split is 89.7% fail / 10.3%
  yellow_card). `bad_touch` shows the same mistake at smaller scale (12.4%
  predicted success vs. 0% true — a mishit can't "succeed").
- **True rare outcomes never predicted**: `keeper_save`/`keeper_punch` are
  ~17-18% `fail` in truth, 0% predicted (model treats every keeper
  save/punch as a clean success). `freekick_crossed` → `offside` (1.1%
  true) never predicted.
- **Card containment**: true cards are 100% contained within `foul` (93/93
  in val) — the data itself follows the domain rule. The model predicts a
  card in only 1/906 foul rows — it "respects" the containment rule only
  because it almost never predicts cards *anywhere*, not because it learned
  conditional structure. ~1% recall on cards.
- **General calibration bias**: the model over-predicts `success` broadly
  (aggregate 90.9% vs. true 82.7%). Biggest gaps: `take_on` (+30pp),
  `corner_crossed` (+28pp), `interception` (+24pp), `keeper_punch`/
  `keeper_save` (+17-18pp), `pass` (+16pp). Two events invert the bias
  instead (`tackle` −21pp, `shot_penalty` −36pp on an 11-example val
  support — likely noise given how unlearned penalties are elsewhere).
- **`red_card`** never appears in any val-side table: 26 true occurrences
  exist in train (step==6 rows) but **0 in the 38-game val split** — not a
  modeling failure, just a rare class landing at zero by which games ended
  up in val. Even in train, 26 examples is in the same "too rare to learn"
  range as `shot_penalty`.

## Root cause synthesis: data scarcity vs. model/representation limits

**Bucket 1 — genuinely not enough data.** `shot_penalty` (73 train),
`shot_freekick` (323), `red_card` (26), and the broader long tail of `event`
types with n < ~500. No architecture change fixes this alone; needs more
matches or careful synthetic augmentation.

**Bucket 2 — model/representation limits, present even with plenty of
data.** At least three separable causes, not one:

1. **Tokenization**: `event`/`result`/`is_home` fed as raw scalars, not
   one-hot/embedded — the model must learn that adjacent category codes
   aren't numerically related. Flagged as an open improvement in
   `mlp.ipynb` already.
2. **Shared 107-wide vocabulary**: class value `1` means `home`/`pass`/
   `time_elapsed=1`/`duration=1` simultaneously, so frequency from one field
   bleeds into unrelated fields. Directly explains the systematic
   over-prediction of `success` across `foul`, `bad_touch`,
   `corner_crossed`, `interception`, `keeper_save`/`keeper_punch`,
   `take_on` — `success` is the single most globally common class and acts
   as an attractive fallback. `tackle`/`take_on`/`interception` already
   have 1,000+ train examples each and still show 20-30pp calibration gaps,
   so more data on those specific events wouldn't fix this on its own.
3. **Feature scale / training shortcuts**: `half_end`/`full_end` had 304
   clean examples each (not data-starved) and a clean available separating
   signal (`minute`/`period_id`), but the model still collapsed them into
   `kick_off` rather than using it — a cheaper shortcut solution, likely
   aggravated by `minute` being unnormalized on a 0-101 scale next to small
   integer category codes.

## Sanity checks performed

- **Overfit 2 training rows**: separate parameters, full-batch SGD (lr=1.5),
  300,000 steps → loss `1.8e-8`, 100% train accuracy, predicted probability
  on true class rounds to 1.0 for both rows. Confirms forward/backward/
  update logic is correctly wired. (Literal float `0.0` loss is not
  practically reachable via plain SGD on `BCEWithLogits` — the loss decays
  roughly like `1/step` past saturation, so hitting true float32 zero this
  way would take on the order of 10^30 steps; not attempted further.)

## Key takeaways

- The model clearly learns from input (43% lower loss than the zeroed
  floor) and shows real, well-calibrated use of context on `is_home` and
  `start_x`/`start_y`.
- Extra depth/capacity (vs. `mlp.ipynb`'s 2-layer model) measurably reduced
  class collapse on `event`/`result`, but didn't eliminate it — it's a
  "use the top few classes well" model, not a "use the whole vocabulary"
  model. Macro precision/recall on `event` (0.29/0.19) is far below micro
  accuracy (0.71).
- Failures split cleanly into two categories needing different fixes: pure
  data scarcity (penalties, red cards, freekicks) vs. representation/shared-
  vocabulary limits that persist even with hundreds to thousands of
  examples (foul/bad_touch predicted as "success", half_end/full_end
  confused with kick_off, systematic success-rate over-prediction).
- The shared 107-wide vocabulary looks like the single highest-leverage fix
  for bucket 2 — it's implicated in the baseline's wrong-constant
  convergence, the `event`/`result` tail collapse, and the broad
  success-rate over-prediction bias, all via the same mechanism (global
  class frequency bleeding across unrelated fields).
