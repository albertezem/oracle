# Oracle

A foundation model for soccer: an autoregressive, decoder-only transformer trained on tokenized match event sequences, built nanoGPT-style.

## Goals

1. Learn transformer/LLM architecture and implementation in depth — attention, embeddings, tokenization, training loops — by building one from scratch.
2. Produce a native soccer event foundation model, following the approach of the reference papers below.

## Reference papers

- **arXiv 2407.14558**, "A Foundation Model for Soccer" — decoder-only transformer trained on StatsBomb open data (FA WSL, ~940K actions across 326 matches), using SPADL actions tokenized as `(team, action type, 10x10 grid cell)`. Small models (365K–548K params).
- **arXiv 2512.17266**, "EventGPT" — player-conditioned, value-aware GPT (nanoGPT-based) trained on 5 seasons of Premier League events (~1,900 matches). Multi-token events: `(team, action type, discretized x/y, elapsed time, outcome, rOBV)` sharing a unified vocabulary, with episode context blocks encoding player IDs and match state. Supports counterfactual player-swap simulation.
- Mendes-Neves, Meireles, Mendes-Moreira, ["A Scalable Approach for Unified Large Events Models in Soccer"](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/ads/preprint_ecml_pkdd_2025_ads_1275.pdf) (ECML PKDD 2025, Applied Data Science track) — replaces the original chain-of-classifiers LEM with one unified tabular autoregressive model. Uses causal masking to sequentially predict each element of the next event (type, accuracy, goal outcome, time elapsed, x/y location, home/away) so the model can condition on the other elements of the same event, not just prior events. This is the paper motivating the current MLP next-event-prediction work (`statsbombpy_test.ipynb`, `socceractiontest.ipynb`).

## Data

- Source: [StatsBomb open data](https://github.com/statsbomb/open-data) — free raw JSON event data (competitions, matches, events, lineups).
- StatsBomb event data format spec: `Open Data Events v4.0.0.pdf` (in repo root).
- Loaders available in the venv: `statsbombpy` (returns pandas DataFrames), `socceraction` (implements SPADL, the flattened action representation paper 1 uses; docs: https://socceraction.readthedocs.io/en/latest/).
- Open design question, not yet decided: raw StatsBomb events (~35 rich, type-specific event types) vs. SPADL (~23 uniform on-ball actions, discards off-ball events like Pressure). This is a research decision to make after exploring real match data, not before.

## Tokenization design decisions

- In the LEM-style 6-field event tuple `(h, e, x, y, t, a)` (paper 3), `t` is defined as **event duration** (how long the action itself took), not time elapsed since the previous event as in the paper. Deliberate deviation — decided when building the masked next-event-prediction table in `stage_to_df.ipynb`.
- Vocab size for `x`/`y` is **105**, not the paper's 101 — matches SPADL's pitch representation of 105m x 68m (one token per meter along the pitch length).

## Environment

- Python 3.12 venv at `.venv/`, created with `uv venv --python 3.12` (system Python was 3.13, which is incompatible with socceraction's constraints).
- Known dependency landmine: `pandera==0.17.2` (pulled in transitively by `socceraction`) declares `multimethod` with no upper bound. Installing latest `multimethod` (2.x) breaks pandera's `from multimethod import ... overload` import. Fix: pin `multimethod<2.0`.
- No `requirements.txt`/`pyproject.toml` committed yet.

## Working relationship — read this before writing any code

The user is the **learner and researcher** on this project; Claude is **supervisor and engineer**, not the implementer.

- The user wants to personally write the granular functions and make the design choices (tokenization scheme, architecture, data representation, training setup, etc.).
- Claude should generate code directly only when the user clearly already understands the concept involved, or for pure boilerplate/infra with no design content (e.g. venv setup, `.gitignore`, dependency debugging).
- **Whenever a medium-or-larger decision comes up, back off** — don't decide it. Either point the user at what they need to learn/read, or hand it back for them to implement, laying out the options and trade-offs rather than picking one.
- Explaining concepts, reviewing the user's code, and debugging environment/infra issues are all in scope for Claude to just do.
- When Claude does write or edit code, match the user's existing coding style (naming, formatting, structure) rather than imposing a different convention.
