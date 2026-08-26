# MLP Evaluation Summary

Results for the trained next-event-prediction MLP (`mlp.ipynb`), evaluated against a zeroed-input baseline and a small human baseline, using the analysis built out in `explore_stage.ipynb`.

## Setup

- **Architecture**: 2-layer MLP, 37 input features → 80 hidden (ReLU) → 107 output logits, trained with `binary_cross_entropy_with_logits` over a **shared 107-wide vocabulary** — all 7 event fields (`is_home`, `event`, `start_x`, `start_y`, `time_elapsed`, `duration`, `result`) share the same output/class-index space rather than having field-specific heads.
- **Training**: 1,000,000 steps, batch size 512, seed 42 throughout (weight init, minibatch sampling, train/val/test split all reproducible). LR schedule: `0.1` (steps 0–500K) → `0.01` (500K–850K) → `0.001` (850K–1M).
- **Data**: SPADL-tokenized StatsBomb events, LEM-style `(h, e, x, y, t, d, a)` tuple per event, split 80/10/10 by game (304/38/38 games; ~74,489 clean val events used below).
- **Zeroed-input baseline**: identical architecture and training procedure, but every input feature is forced to zero on every step. Since gradients w.r.t. the input weights are then always exactly zero, this model is mathematically nothing but a learned bias vector — a clean floor for "did the real model learn anything from the input at all."
- **Human baseline**: one person, 10 examples, self-reported scores from a manual prediction quiz (`explore_stage.ipynb`). Too small a sample to be a rigorous comparison — included as a rough reference point only.

## Loss

Full-dataset (not minibatch-sampled) binary cross-entropy:

| | train | val |
|---|---|---|
| zeroed-input baseline | 0.0412 | 0.0411 |
| **trained model** | **0.023896** | **0.023910** |

The trained model's loss is ~42% lower than the floor — solid evidence it's learning from the input, not just memorizing target frequencies.

## Headline comparison: human vs. baseline vs. trained model

Autoregressive full-event prediction (model predicts all 7 fields from scratch, conditioning on its own earlier guesses within the event), val split, n=74,489 clean events (human: n=10, drawn from the full dataset not the val split specifically):

| field | metric | human (n=10) | zeroed baseline | trained model |
|---|---|---|---|---|
| loss | BCE | — | 0.0411 | 0.0239 |
| is_home | accuracy | 100% | 51.8% | 80.8% |
| event | accuracy | 50% | 43.0% | 65.4% |
| result | accuracy | 50% | 16.9% | 82.8% |
| start_x | MAE | 16.5* | 53.1 | 12.0 |
| start_y | MAE | 16.5* | 32.9 | 9.6 |
| time_elapsed | MAE | 4.6 | 2.20 | 1.98 |
| duration | MAE | 0.6 | 0.897 | 0.918 |

\*human x/y reported as one combined figure, not split.

Notable: `duration` is the one field where the trained model is *not* clearly better than the naive constant-guess baseline — see "Duration: error compounding" below for why.

## Single-masked-field accuracy/MAE (given true context, val)

A different, easier task than the table above: only one field is hidden at a time, and the true values of everything else (including the rest of the current event) are given. Useful for isolating how well the model can use a field's signal when it has perfect information about everything else.

| field | metric | value |
|---|---|---|
| is_home | accuracy | 80.7% |
| event | accuracy | 65.3% |
| result | accuracy | 82.7% |
| start_x | MAE | 12.29 |
| start_y | MAE | 10.44 |
| time_elapsed | MAE | 1.99 |
| duration | MAE | 0.82 |

These track closely with the autoregressive numbers — the model isn't losing much from having to condition on its own guesses, except where noted below.

## Categorical field collapse

**Does the model actually use the full space of categories, or does it default to whatever's most common?** Checked for all three categorical fields (`is_home`, `event`, `result`).

### `event` — collapsed to ~2 classes

Per-class accuracy (recall — of events truly type X, how often the model called it X), autoregressive, val:

| event type | n | zeroed baseline | trained model |
|---|---|---|---|
| pass | 32,055 | 100.0% | 85.0% |
| dribble | 27,888 | 0.0% | 76.7% |
| clearance | 2,230 | 0.0% | 5.0% |
| throw_in | 1,768 | 0.0% | 0.06% |
| **all other 17 types** | 1,524 down to 11 each | 0.0% | **0%** |

Predicted vs. true occurrence rate:

| event type | true rate | trained predicted rate |
|---|---|---|
| pass | 43.0% | 59.3% |
| dribble | 37.4% | 39.9% |
| clearance | 3.0% | 0.58% |
| throw_in | 2.4% | 0.18% |
| **all other 17 types** | 0.01%–2.0% each | **~0%** |

The trained model's predicted vocabulary is essentially `{pass, dribble}` — those two account for 99.3% of everything it ever outputs, regardless of what's actually true. It's a real improvement over the baseline (which only ever says "pass"), but the improvement is entirely about splitting pass vs. dribble correctly; it never learned to use the other 19 categories at all.

### `result` — collapsed even more severely

| result | true rate | baseline predicted | trained predicted | trained accuracy |
|---|---|---|---|---|
| success | 82.7% | 0% | 99.95% | 99.97% |
| fail | 16.9% | 100% | 0.05% | 0.16% |
| offside | 0.18% | 0% | 0% | 0% |
| yellow_card | 0.12% | 0% | 0% | 0% |
| owngoal | 0.004% | 0% | 0% | 0% |
| red_card | 0.000% | 0% | 0% | — |

The trained model has essentially become a constant "always predict success" classifier. Its 82.8% headline accuracy on `result` is almost entirely explained by success being 82.7% of the data — not by learning when a failure/card/offside actually happens. Baseline made the same kind of degenerate bet, just on the wrong constant (`fail`) due to the shared-vocabulary confound (see below).

### `is_home` — NOT collapsed

| is_home | true rate | trained predicted rate | trained accuracy |
|---|---|---|---|
| home | 51.8% | 50.5% | 80.2% |
| away | 48.2% | 49.5% | 81.5% |

Balanced in both prediction rate and accuracy. The collapse pattern seen in `event`/`result` doesn't apply here — likely because `is_home` has only 2 roughly-balanced classes and no long tail to ignore.

## Numeric field coverage

Does the model spread predictions across the full observed range, or cluster narrowly?

| field | true unique values | trained unique values | true std | trained std |
|---|---|---|---|---|
| start_x | 106 | 61 (58%) | 25.0 | 21.6 |
| start_y | 69 | 56 (81%) | 20.2 | 22.4 |
| time_elapsed | 101 | **14 (14%)** | 6.69 | **1.24** |
| duration | 26 | **7 (27%)** | 1.43 | **0.32** |

- **`start_x`/`start_y`**: real coverage, not collapsed — 58–81% of observed values used, and `start_y`'s spread is even wider than the true distribution. The histogram shape is uneven (a handful of sharp "attractor" spikes rather than a smooth match to the true shape), but the model is not ignoring the range.
- **`time_elapsed`/`duration`**: same collapse pattern as `event`/`result`, expressed as clustering tightly around the mode instead of ignoring rare classes outright. Both fields converge to "always guess ~1" and barely deviate.

## Notable behavioral findings

**1. `is_home` after a failed action — a real, strong pattern the model underuses.**
Empirically, the team possessing the ball flips 91.4% of the time after a failed action (vs. 5.8% after a success, vs. 20.7% unconditionally) — a near-deterministic rule. The model has `result-1` (the previous event's outcome) directly as an input feature, but its `is_home` accuracy conditioned on that feature tells a different story: 93.9% after a success, only **17.1%** after a failure. It's picked up *some* signal (better than the ~8.6% a "never flip" heuristic would get), but is nowhere near exploiting the ~91% rule fully.

**2. `duration` — a case of autoregressive error compounding.**
Isolating the effect of context quality on the same clean val set:

| condition | duration MAE |
|---|---|
| always guess 1 (constant) | 0.897 |
| model, given the true other fields | **0.827** |
| model, autoregressive (its own earlier guesses) | **0.918** |

The model genuinely beats the naive constant when given ground truth for the rest of the event. But `duration` is the 6th of 7 fields predicted autoregressively — by the time the model reaches it, it's already guessed `event` (only 65% accurate) and other fields itself. Leaning on those self-generated, sometimes-wrong upstream guesses makes the duration prediction *worse* than just ignoring context entirely. Same root model, opposite outcome, depending on whether its inputs are trustworthy.

**3. Zeroed-input baseline convergence reveals the shared-vocabulary confound directly.**
Queried on an all-zero input, the baseline converges to: `is_home=home`, `event=pass`, `start_x=1`, `start_y=1`, `time_elapsed=1`, `duration=1`, `result=fail`. `is_home`/`event`/`time_elapsed`/`duration` match their true per-field mode — but `start_x` (true mode 49), `start_y` (true mode 64), and `result` (true mode is `success`, not `fail`) don't. The reason: class value `1` is the single most common *target value in the entire dataset* once all 7 fields are pooled together (1,135,032 occurrences vs. 990,086 for class `0`), because it's simultaneously the code for `home`, `pass`, `time_elapsed=1`, and `duration=1`. That inflated global frequency bleeds into fields where `1` isn't actually the right answer — exactly the "not using a shared vocabulary" limitation already flagged as a potential improvement in `mlp.ipynb`, now shown concretely.

## Key takeaways

- The model clearly learns from input (42% lower loss than the zeroed floor) and is not simply memorizing marginal frequencies.
- That said, on 3 of 7 fields (`event`, `result`, and to a lesser extent `time_elapsed`/`duration`), it has substantially collapsed toward the 1–2 most common classes/values and essentially ignores the long tail — its headline accuracy numbers are inflated by class imbalance more than they reflect nuanced prediction.
- `is_home` and, to a real extent, `start_x`/`start_y` are the fields where the model is doing genuine, non-degenerate work.
- The shared 107-wide vocabulary is a concrete, demonstrated source of error (baseline's wrong-constant convergence; likely a contributor to the collapse on `event`/`result` too, since rare classes there compete against globally-frequent codes borrowed from other fields).
- The model has real per-field signal it doesn't fully exploit (`is_home` after a failure) and real signal that gets corrupted by its own compounding errors in the autoregressive chain (`duration`) — two different failure modes worth keeping in mind for any architecture follow-up.
