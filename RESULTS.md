# Evaluation Results

*Nova Chess — <https://novachess.ai>*

The metrics below are computed from the raw policy output of each
model — a single forward-pass distribution over legal moves. None
of the numbers involve tree search, MCTS, or lookahead of any
kind. Every metric measures how well each network's policy matches
the move a human actually played.

Architectural note: Nova is a policy-only network — the model has a
single output head producing move probabilities. Both Maia models
are multi-head: Maia-2 has a policy head plus an auxiliary-
information head and a scalar value-regression head (Tang et al.,
2024); Maia-3 has a policy head plus a 3-dim W/D/L value head
(verified from the published `maia3_simplified.onnx` checkpoint).
The Maia value and auxiliary heads are auxiliary training signals
that were not consulted when computing any metric in this
document — the comparison is strictly on each model's move-policy
output.

All numbers in this document are from a single evaluation run on
**600,000 positions** drawn from Lichess rapid games played in
**March 2026** — stratified at 100,000 positions per rating band.
The March 2026 sample is temporally held out from Nova's training set.

Nova is evaluated in **neutral style** — `classical = 0.5`,
`aggression = 0.5` — for all numbers below. This is the style-matched
comparison against Maia-2 and Maia-3, which do not have style
conditioning. A separate "actual-style" Nova run that uses each
player's measured style is shipped as `nova_actual_600k.pkl`; on
aggregate it moves the numbers by less than 0.1 percentage point, so
style conditioning does not materially affect aggregate results on
this sample.


## Model checkpoints used

- **Nova**: the checkpoint published in this repo / on Hugging Face
  ONNX export.
- **Maia-2**: the `rapid_model.pt` checkpoint published by the Maia
  team (23.3M parameters).
- **Maia-3**: the `maia3_simplified.onnx` checkpoint — the publicly
  available Maia-3 checkpoint we evaluated, published on
  https://maiachess.com. All Maia-3 numbers in this document refer
  specifically to this checkpoint.


## Headline

All four metrics are measured on the same 600K sample for each of
Maia-2, Maia-3, and Nova:

| Metric | Maia-2 | Maia-3 | **Nova** | Winner (Nova vs Maia-3) |
|---|---|---|---|---|
| Top-1 hit rate | 50.27 % | **54.83 %** | 54.60 % | Maia-3 by 0.23 pp |
| Top-5 hit rate | 88.38 % | **91.23 %** | 91.10 % | Maia-3 by 0.13 pp |
| Mean P(actual move) | 38.44 % | 42.10 % | **42.51 %** | Nova by 0.41 pp |
| Mean top-5 probability mass | 89.33 % | 91.96 % | **92.26 %** | Nova by 0.30 pp |

Maia-3 is ahead on hit1 and hit5 by 0.23 pp and 0.13 pp. Nova is
ahead on Mean P(actual) and Mean top-5 mass by 0.42 pp and 0.30 pp.
All four deltas persist on the sub-filtered variants (ply ≥ 10;
ply ≥ 10 + clk ≥ 30s) with the same direction.

The split between hit rate (Maia-3 ahead) and probability mass
(Nova ahead) is a ranking-vs-calibration tradeoff: the two models
are comparably accurate on top-1 argmax, and Nova places on average
more probability mass on the move the human actually played.

Per-filter deltas (Nova − Maia-3, same samples as tables above):

| Filter | hit1 | hit5 | Mean P(actual) | Mean top-5 mass |
|---|---|---|---|---|
| all | −0.23 pp | −0.13 pp | +0.42 pp | +0.30 pp |
| ply ≥ 10 | −0.34 pp | −0.14 pp | +0.31 pp | +0.42 pp |
| ply ≥ 10 + clk ≥ 30s | −0.37 pp | −0.17 pp | +0.26 pp | +0.40 pp |

All four signs are consistent across filters.

Paired significance tests on the 600,000-position `all` filter:

| Metric | Delta | Test | p-value |
|---|---|---|---|
| hit1 | −0.23 pp | McNemar, exact binomial on 69,149 discordant pairs | p < 10⁻⁶ |
| hit5 | −0.13 pp | McNemar, exact binomial on 25,603 discordant pairs | p < 10⁻⁵ |
| Mean P(actual) | +0.42 pp | paired t-test (per-position) | p < 10⁻⁵ |
| Mean top-5 mass | +0.30 pp | paired t-test (per-position) | p < 10⁻⁵ |

Both probability-mass deltas were also re-estimated via a
player-clustered bootstrap (2,000 resamples, resampling at
`player_id` level to account for within-player correlation):

- **Mean P(actual)**: 95 % CI [+0.38, +0.46] pp, p < 0.001
- **Mean top-5 mass**: 95 % CI [+0.28, +0.31] pp, p < 0.001

Clustered and unclustered intervals are very close at this sample
size.

Maia-2 is included as a reference baseline. It trails Maia-3 and
Nova by about 4.6 pp on hit1 and 3–4 pp on Mean P(actual).


## Overall — three filter variants

Three filters, each for a different reason:

- **all** — every position in the sample.
- **ply ≥ 10** — excludes opening-theory positions.
- **ply ≥ 10 + clk ≥ 30s** — also excludes time-scramble positions.

### Hit rate

| Filter | n | Metric | Maia-2 | Maia-3 | Nova |
|---|---|---|---|---|---|
| all | 600,000 | hit1 | 50.27 | **54.83** | 54.60 |
| all | 600,000 | hit5 | 88.38 | **91.23** | 91.10 |
| ply ≥ 10 | 512,234 | hit1 | 52.89 | **55.61** | 55.27 |
| ply ≥ 10 | 512,234 | hit5 | 89.91 | **91.43** | 91.29 |
| ply ≥ 10 + clk ≥ 30s | 489,141 | hit1 | 52.80 | **55.51** | 55.14 |
| ply ≥ 10 + clk ≥ 30s | 489,141 | hit5 | 89.88 | **91.39** | 91.22 |

### Probability mass

| Filter | n | Metric | Maia-2 | Maia-3 | Nova |
|---|---|---|---|---|---|
| all | 600,000 | Mean P(actual) | 38.44 | 42.10 | **42.51** |
| all | 600,000 | Mean top-5 mass | 89.33 | 91.96 | **92.26** |
| ply ≥ 10 | 512,234 | Mean P(actual) | 40.89 | 43.31 | **43.62** |
| ply ≥ 10 | 512,234 | Mean top-5 mass | 90.61 | 92.17 | **92.59** |
| ply ≥ 10 + clk ≥ 30s | 489,141 | Mean P(actual) | 40.75 | 43.14 | **43.40** |
| ply ≥ 10 + clk ≥ 30s | 489,141 | Mean top-5 mass | 90.48 | 92.04 | **92.44** |

Across all three filters: Maia-3 is ahead on hit1 and hit5;
Nova is ahead on Mean P(actual) and Mean top-5 mass.


## Per-rating-band — hit1 (all positions)

Each band has exactly 100,000 positions.

| Band | Maia-2 | Maia-3 | Nova | Nova − Maia-3 |
|---|---|---|---|---|
| 800–1100 | 45.37 | 49.08 | **49.30** | **+0.23** |
| 1100–1400 | 48.89 | 51.80 | **52.34** | **+0.53** |
| 1400–1700 | 50.92 | 54.09 | **54.10** | +0.00 |
| 1700–2000 | 52.04 | **56.15** | 55.81 | −0.34 |
| 2000–2300 | 52.48 | **57.92** | 57.49 | −0.43 |
| 2300–4000 | 51.91 | **59.92** | 58.55 | −1.37 |
| **UW-6band** | 50.27 | **54.83** | 54.60 | −0.23 |

Nova leads Maia-3 in the 800–1100 and 1100–1400 bands, ties at
1400–1700, and trails in the three bands above 1700. The 2300–4000
band is the largest Nova-vs-Maia-3 gap in the table at −1.37 pp.


## Per-rating-band — Mean P(actual) (all positions)

The same slice, measured by how much probability mass each model put
on the move that was actually played.

| Band | Maia-2 | Maia-3 | Nova | Nova − Maia-3 |
|---|---|---|---|---|
| 800–1100 | 33.45 | 36.00 | **36.95** | **+0.95** |
| 1100–1400 | 36.92 | 38.94 | **39.65** | **+0.71** |
| 1400–1700 | 38.86 | 41.12 | **41.55** | **+0.43** |
| 1700–2000 | 40.24 | 43.30 | **43.61** | **+0.31** |
| 2000–2300 | 40.91 | 45.61 | **45.95** | **+0.34** |
| 2300–4000 | 40.24 | **47.60** | 47.38 | −0.22 |
| **UW-6band** | 38.44 | 42.10 | **42.51** | **+0.41** |

Nova's Mean P(actual) is higher than Maia-3's in five of six bands;
Maia-3 leads in the 2300–4000 band by 0.22 pp. The Nova-vs-Maia-3
gap is largest in the 800–1100 band (+0.95 pp) and narrows in each
higher band.


## Per-Maia-tier

Tiers: Skilled (800–1599), Advanced (1600–1999), Master (2000+).

### Hit1

| Tier | Band range | Maia-2 | Maia-3 | Nova | Nova − Maia-3 |
|---|---|---|---|---|---|
| Skilled | 800–1599 | 47.97 | 51.24 | **51.51** | **+0.27** |
| Advanced | 1600–1999 | 51.88 | **55.74** | 55.53 | −0.21 |
| Master | 2000+ | 52.20 | **58.92** | 58.02 | −0.90 |
| **UW-tier avg** | — | 50.68 | **55.30** | 55.02 | −0.28 |

### Mean P(actual)

| Tier | Maia-2 | Maia-3 | Nova | Nova − Maia-3 |
|---|---|---|---|---|
| Skilled | 35.97 | 38.25 | **38.98** | **+0.73** |
| Advanced | 40.06 | 42.91 | **43.26** | **+0.35** |
| Master | 40.57 | 46.61 | **46.67** | **+0.06** |
| **UW-tier avg** | 38.87 | 42.59 | **42.97** | **+0.38** |

Nova's Mean P(actual) is higher than Maia-3's in all three tiers
(+0.73, +0.35, +0.06). On hit1 Nova leads the Skilled tier by 0.27
pp and trails in the other two tiers.


## Per-phase (by move number, all positions)

### Hit1

| Phase | n | Maia-2 | Maia-3 | Nova |
|---|---|---|---|---|
| M1–5 (early opening) | 87,766 | 34.97 | 50.27 | **50.69** |
| M6–10 | 85,694 | 50.98 | **52.12** | 52.02 |
| M11–20 | 154,592 | 51.38 | **53.90** | 53.15 |
| M21–35 | 157,099 | 52.91 | **56.44** | 55.75 |
| M36–50 | 74,912 | 56.14 | **59.47** | 59.37 |
| M51+ | 39,937 | 56.65 | 59.18 | **60.86** |

### Mean P(actual)

| Phase | Maia-2 | Maia-3 | Nova |
|---|---|---|---|
| M1–5 | 24.16 | 35.03 | **36.04** |
| M6–10 | 38.37 | **39.67** | 39.53 |
| M11–20 | 39.14 | **41.32** | 41.11 |
| M21–35 | 41.11 | 44.04 | **44.26** |
| M36–50 | 44.31 | 47.41 | **48.35** |
| M51+ | 45.77 | 48.23 | **50.77** |

**M1–5.** Maia-2 hit1 is 34.97 % and Maia-2 Mean P(actual) is
24.16 %; Maia-3 and Nova both land near 50 % on hit1 and near 35 %
on Mean P. In this phase Nova leads Maia-3 by +0.42 pp on hit1 and
+1.01 pp on Mean P.

**M6–10, M11–20, M21–35.** Maia-3 leads on hit1 in all three buckets
(+0.10, +0.75, +0.69 pp). On Mean P, Maia-3 leads in M6–10 and
M11–20 (+0.14, +0.21 pp); Nova leads in M21–35 (+0.22 pp).

**M36–50.** Maia-3 leads hit1 by 0.10 pp; Nova leads Mean P by 0.94 pp.

**M51+.** Nova leads hit1 by 1.68 pp and Mean P by 2.54 pp.


## Per piece count (all positions)

### Hit1

| Piece count | n | Maia-2 | Maia-3 | Nova |
|---|---|---|---|---|
| 28–32 (opening) | 226,252 | 44.00 | **51.06** | 50.91 |
| 24–27 | 96,499 | 52.14 | **54.80** | 54.13 |
| 20–23 | 83,496 | 52.79 | **56.17** | 55.42 |
| 16–19 | 69,487 | 54.02 | **57.40** | 56.97 |
| 12–15 | 56,412 | 55.42 | **58.57** | 58.26 |
| 8–11 | 40,477 | 56.19 | 59.33 | **59.80** |
| 3–7 (late endgame) | 27,377 | 58.84 | 61.14 | **62.99** |

### Mean P(actual)

| Piece count | Maia-2 | Maia-3 | Nova |
|---|---|---|---|
| 28–32 | 32.11 | 37.36 | **37.62** |
| 24–27 | 40.04 | 42.38 | 42.28 |
| 20–23 | 40.83 | 43.64 | **43.69** |
| 16–19 | 42.25 | 45.21 | **45.52** |
| 12–15 | 43.75 | 46.71 | **47.32** |
| 8–11 | 44.57 | 47.50 | **48.90** |
| 3–7 | 48.15 | 50.10 | **53.24** |

Nova's Mean P(actual) is higher than Maia-3's in six of seven
piece-count buckets (all but 24–27). Nova's hit1 is higher than
Maia-3's in the two lowest buckets (8–11, 3–7). The Nova-vs-Maia-3
Mean P gaps in those two buckets are +1.40 pp and +3.14 pp.


## Deltas — Nova vs. Maia-3, per band, per metric (all filter)

Nova − Maia-3 in percentage points.

| Band | hit1 | hit5 | Mean P(actual) | Mean top-5 mass |
|---|---|---|---|---|
| 800–1100 | +0.23 | −0.07 | +0.95 | +0.85 |
| 1100–1400 | +0.53 | −0.16 | +0.71 | +0.28 |
| 1400–1700 | +0.00 | −0.11 | +0.43 | +0.14 |
| 1700–2000 | −0.34 | −0.13 | +0.31 | +0.17 |
| 2000–2300 | −0.43 | −0.14 | +0.34 | +0.28 |
| 2300–4000 | −1.37 | −0.06 | −0.22 | +0.04 |
| **UW-6band** | −0.23 | −0.11 | **+0.41** | **+0.30** |

Per-band direction of the delta:
- hit1: Nova ahead in 2 bands, tied in 1, Maia-3 ahead in 3.
- hit5: Maia-3 ahead in all 6 bands.
- Mean P(actual): Nova ahead in 5 bands, Maia-3 ahead in 1.
- Mean top-5 mass: Nova ahead in all 6 bands.


## Summary of the four overall deltas

On the full 600K sample:

- **hit1**: Maia-3 leads by 0.23 pp (54.83 vs 54.60).
- **hit5**: Maia-3 leads by 0.13 pp (91.23 vs 91.10).
- **Mean P(actual)**: Nova leads by 0.41 pp (42.51 vs 42.10), i.e.,
  Nova assigns on average 0.41 pp more probability to the move the
  human played.
- **Mean top-5 mass**: Nova leads by 0.30 pp (92.26 vs 91.96), i.e.,
  Nova's top-5 predictions carry on average 0.30 pp more total
  probability.

hit1 and hit5 are argmax metrics (is the top / top-5 prediction
correct). Mean P(actual) and Mean top-5 mass are probability-mass
metrics (how much mass is assigned to the actual move / to the top
5). Argmax metrics and probability-mass metrics need not rank
models the same way.


## Actual-style vs neutral-style Nova

Nova conditions inference on three inputs:

- `rating` ∈ [800, 2700]
- `classical` ∈ [0, 1]
- `aggression` ∈ [0, 1]

All numbers in this document use `classical = 0.5, aggression = 0.5`
("neutral style"). A separate run conditions Nova on each position's
player-measured classical and aggression values ("actual style"). On
the same 600K sample the two runs differ as follows (overall, all
filter):

| | Neutral | Actual |
|---|---|---|
| hit1 | 54.60 | 54.52 |
| hit5 | 91.10 | 91.04 |
| Mean P(actual) | 42.51 | 42.45 |
| Mean top-5 mass | 92.26 | 92.14 |

The actual-style run differs from neutral by ≤ 0.12 pp on every
aggregate metric in this sample.


## Reproducing these numbers

The 600,000-position evaluation sample (`unified_sample_600k.pkl`)
and the Nova model weights (`nova_v3b.onnx` + `nova_v3b.onnx.data`)
are published on Hugging Face:

<https://huggingface.co/novachess/novachess-engine>

The sample is a pandas DataFrame with one row per position. Key
columns:

- `fen` — position in Forsyth-Edwards notation
- `actual` — the move the human actually played (UCI)
- `rating` — the player's rating at the time of the game
- `band` — rating band bucket (`800-1100`, `1100-1400`, …, `2300-4000`)
- `ply` — ply number within the game (0-indexed)
- `min_clock` — minimum clock time remaining of the two sides, in seconds
- `piece_count` — number of pieces on the board
- `player_id` — integer player ID (used for the clustered bootstrap in
  the significance tests)

Any inference pipeline that loads the ONNX model, encodes each FEN
into the 18-plane input format described in the model card, feeds
`(rating_norm, 0.5, 0.5)` as conditioning (for neutral-style), and
records top-1/top-5/probability-mass metrics can reproduce the
tables above.

For convenience, the per-model prediction outputs on the same 600K
sample are also published on Hugging Face:

- `nova_neutral_600k.pkl` — Nova at `(rating, 0.5, 0.5)` per position
- `nova_actual_600k.pkl` — Nova at each player's measured style
- `maia2_600k.pkl` — Maia-2 (`rapid_model.pt`) predictions
- `maia3_600k.pkl` — Maia-3 (`maia3_simplified.onnx`) predictions

Each pickle is a pandas DataFrame aligned 1:1 with `unified_sample_600k.pkl`
and contains the per-position policy distribution. Loading these
sidesteps the need to re-run inference, and lets anyone verify our
hit-rate / probability-mass tables and paired-significance numbers
directly.

### Verifying the headline numbers

Each prediction pickle has the per-position metrics pre-computed as
columns, so the four headline numbers can be reproduced in five
lines from the Nova pkl alone:

```python
import pandas as pd
preds = pd.read_pickle("nova_neutral_600k.pkl")

print(f"hit1            = {preds['hit1'].mean()*100:.2f}%   (paper: 54.60%)")
print(f"hit5            = {preds['hit5'].mean()*100:.2f}%   (paper: 91.10%)")
print(f"Mean P(actual)  = {preds['actual_prob'].mean()*100:.2f}%   (paper: 42.51%)")
print(f"Mean top-5 mass = {preds['top5_probs'].apply(sum).mean()*100:.2f}%   (paper: 92.26%)")
```

Runs in seconds — no inference, no per-row loop. Substitute
`maia2_600k.pkl` or `maia3_600k.pkl` for `nova_neutral_600k.pkl` to
verify the corresponding rows of the headline table; both follow the
same column schema.

Per-position columns in the prediction pickles (relevant subset):

- `predicted` — top-1 predicted move (UCI string, legal-masked argmax)
- `top5` — list of top-5 UCI strings
- `top5_probs` — list of top-5 probabilities aligned with `top5`
- `actual_prob` — probability the model assigned to `actual`
- `hit1` / `hit5` — booleans (predicted == actual / actual ∈ top5)
- `style_used` / `style_mode` — `(rating_norm, classical, aggression)`
  the model was conditioned on, and a label like `"neutral"` /
  `"actual"` (Maia pkls do not have these columns)

For per-band, per-tier, per-phase, or per-piece-count breakdowns,
group on the corresponding column from `unified_sample_600k.pkl`
(which is aligned 1:1 by row index) — e.g.:

```python
import pandas as pd
sample = pd.read_pickle("unified_sample_600k.pkl")
preds  = pd.read_pickle("nova_neutral_600k.pkl")

# Per-rating-band hit1 (matches the per-band table above)
print(preds.assign(band=sample["band"].values)
            .groupby("band")["hit1"].mean()
            .mul(100).round(2))
```

To regenerate the Maia predictions from scratch instead, download
their public checkpoints from <https://maiachess.com> and run them
on the same `unified_sample_600k.pkl` file.

---

Nova Chess — <https://novachess.ai>
