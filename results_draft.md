# Results (draft for write-up)

All numbers come from a full re-execution of notebooks 00–14 (commit
`0ad4084`). Splits and training are seeded.

## Setup

- **Data:** StatsBomb open data, Premier League 2015/16, 380 matches, as SPADL
  actions. Each event is a 7-field tuple
  `(is_home, event_type, start_x, start_y, time_elapsed, duration, result)`.
  Each event becomes 7 rows, one per field, with that field and the ones after
  it masked.
- **Split:** 80/10/10 by game (seed 42). Train 4,150,517 rows, val 523,817
  rows (74,831 events), test 517,216 rows (38 games).
- **Input:** the 3 previous events plus match state (period, minute, second,
  goal and card counts).
- **Metrics:** accuracy for `is_home` / `event` / `result`, mean absolute
  error (MAE) for the other fields. x/y are in metres on a 105 × 68 pitch,
  time and duration in seconds.
- **Two evaluation modes:**
  - *Teacher-forced (TF):* the true values of the earlier fields in the same
    event are given.
  - *Autoregressive (AR):* the model fills in fields one at a time, using its
    own predictions for the earlier ones.
  - Predictions are argmax in both modes.
- **Naive baseline:** always predict the training-set mode (categorical
  fields) or median (numeric fields).

## Model progression

Each version changes one thing from the one before.

| | change |
|---|---|
| v1 | 2-layer MLP `37→80→107`, one 107-wide output vocabulary shared by all fields, categories fed as raw numeric codes, BCE loss |
| v2 | deeper `37→96→96→96→107` |
| v3 | same as v2, trained longer, backward pass written by hand |
| v4 | categorical fields through learned embeddings, output vocabulary still shared |
| v5 | 7 separate output heads, one cross-entropy per field; continuous inputs normalised |

Summed per-field cross-entropy on val: v5 **9.51**, against **14.96** for
predicting each field's training distribution.

## Validation results (teacher-forced)

| field | metric | naive | v1 | v2 | v3 | v4 | **v5** |
|---|---|---|---|---|---|---|---|
| is_home | acc | 51.8% | 82.3% | 94.2% | 93.9% | 94.5% | **95.0%** |
| event | acc | 43.0% | 66.1% | 69.8% | 71.0% | 72.4% | **75.5%** |
| result | acc | 82.7% | 82.7% | 86.6% | 85.6% | 86.7% | **88.4%** |
| start_x | MAE | 20.6 | 12.26 | 8.62 | 9.00 | 7.98 | **6.49** |
| start_y | MAE | 17.5 | 10.43 | 8.82 | 9.06 | 8.06 | **6.90** |
| time_elapsed | MAE | 2.19 | 1.99 | 1.67 | 1.74 | 1.60 | **0.91** |
| duration | MAE | 0.90 | 0.82 | 0.87 | 0.81 | 0.79 | **0.75** |

v3 is no better than v2. It has v2's architecture and differs only in the
hand-written backward pass and the longer training, so that is expected. Its
point is that the hand-written backward pass matches autograd.

## Test set (held out, evaluated once)

| field | metric | naive · AR | v5 · AR | naive · TF | v5 · TF |
|---|---|---|---|---|---|
| is_home | acc | 0.516 | **0.949** | 0.516 | **0.949** |
| event | acc | 0.433 | **0.734** | 0.433 | **0.753** |
| start_x | MAE | 20.80 | **6.73** | 20.80 | **6.48** |
| start_y | MAE | 17.57 | **7.41** | 17.57 | **6.87** |
| time_elapsed | MAE | 2.23 | **1.43** | 2.23 | **0.93** |
| duration | MAE | 0.900 | **0.887** | 0.900 | **0.753** |
| result | acc | **0.824** | 0.821 | 0.824 | **0.881** |

- v5 summed cross-entropy on test: **9.575 (TF)** and **11.249 (AR)**.
- v5 beats naive on 7/7 fields teacher-forced and 6/7 autoregressive.
- The autoregressive loss is on `result`. It is predicted last, conditioned on
  six of the model's own guesses, against an "always success" prior that is
  already right 82% of the time.
- Test is within ~0.003 accuracy and ~0.03 MAE of val on every field, so there
  is no sign of overfitting to val.

## Scaling (v5 architecture, 300k steps each, val loss)

| train data | rows | val loss |
|---|---|---|
| 25% | 1.04M | 10.072 |
| 50% | 2.08M | 10.042 |
| 100% | 4.15M | 10.015 |

| hidden layers | params | val loss |
|---|---|---|
| `[80]` | 33.7k | 11.051 |
| `[96, 96]` | 49.7k | 10.444 |
| `[96, 96, 96]` | 59.0k | 10.015 |

- Loss falls steadily with both data and parameters.
- Parameters matter far more at this scale. Quadrupling the data gains 0.06
  nats; going from one hidden layer to three gains 1.04.
- With fixed steps, the model looks capacity-limited rather than data-limited.

## Findings

**1. The shared output vocabulary was the main bottleneck.**
- With one 107-wide output for all fields, class `1` meant `home`, `pass`,
  `time_elapsed=1` and `duration=1` at once, so frequency leaked between fields.
- v3 (shared vocabulary) predicts 18 of 24 event types. It never predicts
  `take_on` (0 recall) and almost never `bad_touch` (4 predictions, all wrong).
  Recall is 0.8% for `shot`, 0.6% for `cross` and 0.2% for `foul`.
- It over-predicts `dribble` (46% predicted vs 37% true).
- It pushes `result` towards `success` (93% predicted vs 83% true).
- Separate per-field heads (v5) improved every field at once, more than
  embeddings (v4) did.

**2. Calibration.**
- v3 is overconfident on the dominant classes: expected calibration error
  (ECE) 0.094 for `pass`, 0.054 for `dribble`.
- v5's ECE is below 0.007 for every class of `is_home`, `event` and `result`.
- Mean probability on the true class rises sharply from v3 to v5 for
  mid-frequency classes: `shot` 0.054 → 0.229, `foul` 0.084 → 0.318,
  `cross` 0.060 → 0.163.
- v5's remaining poorly calibrated classes are ones it almost never assigns
  probability to: `take_on` (0.086), `bad_touch` (0.090), `offside` (0.006),
  `owngoal` (0.006).

**3. Two kinds of failure.**
- Classes with only tens of examples (`shot_penalty`, `owngoal`, `red_card`)
  can't be learned whatever the architecture; they need more matches.
- Once the representation is fixed, structural predictability matters more
  than example count. Near-deterministic events given 3 events of history (set
  pieces, period boundaries) are learned at any support. Open-play actions with
  several plausible continuations (`take_on`, `bad_touch`) stay hard even with
  1,000+ examples.

**4. Human reference (10-example quiz, self-scored, rough only).**
- Human: 100% `is_home`, 50% `event`, 50% `result`, x/y MAE 16.5,
  `time_elapsed` MAE 4.6, `duration` MAE 0.6.

**5. The hand-written backward pass is correct.**
- v3 overfits 2 training rows to a loss of 1.8e-8 with exact predictions.
- On val it lands within ~0.4 MAE / ~1 point accuracy of autograd v2.

## Limitations

- **Full-match simulation (12) doesn't work yet.**
  - Generated matches never produce `full_end`. A simulated match ran to the
    8,000-event cap still in period 1 at "minute 451" (score 19–8).
  - The match clock never closes the half, so the score distribution can't be
    compared with real matches (val: 1.39 home, 1.34 away goals per game).
- **Run-to-run variation.**
  - v4 and v5 reproduce to within 0.01.
  - v1 and v3 moved between runs despite seeding: v3 by up to 0.35 MAE on
    `start_y` and 0.8 points on `result`.
  - Treat the v1–v3 differences as approximate.
- **No team or player identity.** Results are population-level only.
