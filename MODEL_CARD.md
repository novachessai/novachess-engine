---
license: other
license_name: nova-chess-engine-license
license_link: https://github.com/novachessai/novachess-engine/blob/main/LICENSE
language: en
tags:
  - chess
  - transformer
  - human-move-prediction
  - style-conditioned
pipeline_tag: other
library_name: onnx
---

# Nova Chess Engine

A style-conditioned transformer that predicts human chess moves. Given
a board position, a target player rating, and two optional style parameters
(classical–hypermodern preference and aggression), Nova returns a
probability distribution over all legal moves calibrated to how a
player of that rating and style would play.

**Inference is a single forward pass of the neural network.** Nova
does not use Monte Carlo tree search, minimax, alpha-beta pruning,
or any form of engine-style position evaluation. There is no value
head and no lookahead. Move selection comes entirely from learned
patterns over the training corpus — fast on CPU (~35-50 ms per
position) and categorically different from search-based engines
like Stockfish or Leela.

**Play Nova directly at [novachess.ai](https://novachess.ai)** —
Nova powers the *Play* and *Train with Nova* features on the site,
where you can face Nova at any rating/style setting with built-in
analysis and post-game review.

- Developed by Nova Chess — <https://novachess.ai>
- Model type: pure-policy neural network over chess moves,
  conditioned on position + rating + two style parameters.
  Single forward pass per position; no search, no value head, no
  game history.
- Language(s): not applicable — input is chess positions (18-channel
  plane encoding), output is a distribution over 16,384 move indices
- License: custom non-commercial (see `LICENSE`)

Full results and reproducibility: [`RESULTS.md`](RESULTS.md).
Source code and docs: <https://github.com/novachessai/novachess-engine>


## Model Details

### Inputs

- `positions` — `float32` tensor of shape `(B, 18, 8, 8)`
  - Planes 0–5: white pieces (P, N, B, R, Q, K), one-hot per square
  - Planes 6–11: black pieces (P, N, B, R, Q, K)
  - Plane 12: side to move (1s if white to move, 0s otherwise)
  - Planes 13–16: castling rights (white-kingside, white-queenside,
    black-kingside, black-queenside)
  - Plane 17: en-passant file indicator
- `conditioning` — `float32` tensor of shape `(B, 3)`
  - `rating_norm` = `(rating − 800) / (2700 − 800)`, clipped to `[0, 1]`
  - `classical` ∈ `[0, 1]` — opening preference (higher = more
    classical mainlines)
  - `aggression` ∈ `[0, 1]` — tactical/sacrificial tendency

### Outputs

- `logits` — `float32` tensor of shape `(B, 16384)`. Raw logits over
  the 16,384-index move space. Caller is responsible for masking
  illegal moves and applying softmax.

Move index encoding:

```
move_index = promotion_offset + from_square * 64 + to_square
promotion_offset:
    0      no promotion (also queen promotion)
    4096   knight promotion
    8192   bishop promotion
    12288  rook promotion
```

where `from_square` and `to_square` are standard 0–63 indices
(`a1 = 0, h8 = 63`).

### Architectural distinction from prior work

Nova is **single-head pure-policy**: the network's only output is the
move distribution. There is no value head (no game-outcome
prediction), no auxiliary head (no side-task supervision on captures,
checks, etc.), no search at inference, no lookahead, and no use of
any position evaluator before, during, or after the forward pass.
Move selection comes entirely from the policy distribution the
network learned by predicting actual human moves at the conditioned
rating and style.

This contrasts with every published comparable model:

| Model | Heads at inference | Search | Notes |
|---|---|---|---|
| **Nova** (this release) | 1 (policy) | none | single forward pass, ~35–50 ms CPU |
| Maia-2 (NeurIPS 2024) | 3 (policy + value + auxiliary) | none | value head regresses W/D/L; auxiliary head predicts legal moves, captures, check delivery |
| Maia-3 (`maia3_simplified.onnx`) | 2 (policy + value) | none | drops Maia-2's auxiliary head; retains W/D/L value head |
| Allie (ICLR 2025) | 1 (policy) + value via search | adaptive MCTS at inference | policy is decoder-only over move sequences; MCTS provides per-position evaluation at runtime |
| Leela (LC0) | 2 (policy + value) | MCTS | engine-strength playing model |
| Stockfish (NNUE) | evaluation only | alpha-beta | not a human-move predictor |

The pure-policy stance is a deliberate design choice. It keeps the
model fast (one forward pass, no tree expansion), simple to deploy
(just the ONNX file — no MCTS implementation, no auxiliary supervision
data at training time, no value-head calibration to maintain), and
forces the network to learn move quality entirely from move-selection
patterns rather than offloading it to a parallel evaluator. The
benchmarks in `RESULTS.md` show this is competitive with multi-head
architectures on the move-prediction task itself.


### Files in this repository

- `nova.onnx` + `nova.onnx.data` — ONNX export with external
  data. Both files required at inference time; place in the same
  directory before loading.
- `nova.pt` — PyTorch checkpoint (weights only) for research
  and fine-tuning.
- `unified_sample_600k.pkl` — the 600K-position out-of-sample
  evaluation set used in the results reported below. Schema:
  `{fen, actual, rating, ply, min_clock, piece_count, band,
  player_id, result, ...}`.
- `nova_neutral_600k.pkl`, `nova_actual_600k.pkl`,
  `maia2_600k.pkl`, `maia3_600k.pkl` — per-position predictions from
  each model on the 600K sample, used for the paired significance
  tests in `RESULTS.md`.


## Uses

### Direct use

- Predict the probability distribution over legal moves that a human
  of a specified rating and style would play from a given position.
- Sample moves to run as a human-like opponent or study partner.
- Score actual human moves by `P(actual_move | position, rating,
  style)` for humanness analysis, move difficulty assessment, or
  anti-cheat signals.
- Benchmark other human-move predictors against Nova on shared
  evaluation sets.
- Fine-tune on specialized data (specific player corpora, specific
  opening systems, specific time controls) for personal or research
  use.

### Downstream use

The model is used internally by Nova Chess to power the
*Play Nova* and *Train with Nova* features at https://novachess.ai,
where end users play or practice against the model at chosen rating
and style settings. The same weights published in this repository are
the ones served by the application.

The in-app version additionally wraps Nova's policy output with a
small calibration layer used to tune playing strength across rating
tiers. The two main components are a **per-tier temperature schedule**
(the primary calibration lever) and an **evaluation-only filter**:
after Nova samples a candidate move, Stockfish is consulted at low
depth to evaluate that specific candidate; if its evaluation falls
below a tier-dependent quality threshold, the move is probabilistically
replaced by re-sampling from Nova's own distribution.

**Every move the in-app bot plays still originates from Nova's policy
distribution.** Stockfish is never used to suggest, generate, or
select moves — only to evaluate moves Nova has already proposed, so
that obvious blunders at higher tiers can be probabilistically caught.
The model weights are never touched. The calibration layer is **not**
part of this release; the released checkpoint is the bare policy
model, exactly the surface that benchmarks and downstream research /
fine-tuning should target. See the README's "In-app behavior vs the
released model" section for the full distinction.

### Out-of-scope use

- Not suitable as a chess-playing engine for maximum-strength
  competition against search-based engines. Nova is trained on
  human-move prediction — it maximizes `P(move | human of rating R)`,
  not `P(best move)` — and uses no search, no lookahead, and no
  position evaluation beyond the single forward pass of the policy
  network. For engine-quality play, a search-based evaluator such as
  Stockfish remains the correct choice.
- Not intended for cheating detection as a standalone verdict. Nova
  probabilities can inform a cheat-detection pipeline but should not
  be used as sole evidence for accusations.
- Not validated on chess variants (Chess960, King of the Hill, etc.)
  — trained only on standard chess.
- Not a replacement for human coaching — move probabilities are not
  explanations, and the model does not produce commentary or verbal
  analysis.


## Bias, Risks, and Limitations

- **Training-data distribution.** Nova is trained on ~520M positions
  from Lichess rapid games played Apr–Nov 2025. The player population
  is self-selected (online rapid players on one platform), skews
  toward active users in rating bands 1100–2300, and may not
  represent the full distribution of human chess play. Inferences
  about moves at extreme ratings (particularly below 800 and above
  2500) have less training-data support.
- **Style axis limitations.** The classical and aggression axes
  capture specific operational definitions (opening move choices for
  classical; captures + territorial control + king pressure for
  aggression). They do not capture all dimensions of human chess
  style (combinational richness, prophylaxis, time management, etc.).
- **Rating conditioning is a scalar.** Nova receives a single number
  for rating, not a distribution. The model has learned a continuous
  interpolation of playing strength, but at the high end of the
  rating axis the playing strength it produces may saturate below the
  conditioned rating.
- **No game history.** Nova conditions on the current position only,
  not on the preceding move sequence. Two positions with identical
  FENs are indistinguishable to the model even if reached through
  very different games.
- **No check for illegal moves.** The raw logits include mass on
  illegal move indices. Callers must apply a legal-move mask before
  sampling. See the README quickstart and `docs/serving.md` for the
  reference masking pattern.
- **Value / result prediction is not supported.** This checkpoint is
  policy-only; it does not output win/draw/loss probabilities.


## Training Details

### Training data

Nova was trained on a large corpus of Lichess rapid games, balanced
across six rating bands from 800 to 2700+. Position sampling and
filtering were tuned to keep all skill levels and all game phases
well-represented in training. Details of the data pipeline and
cohort balancing are not published.

### Training procedure

Nova is trained end-to-end with a cross-entropy objective over the
16,384-index move space. The output is the policy distribution over
legal moves, with no auxiliary value head. Specific architectural
dimensions and training hyperparameters are not published.

### Inference cost

- CPU (ONNX fp32): 35–50 ms per position on a modern x86 core
- GPU (batched, H100): ~1 ms per position
- Inference memory: approximately 500 MB RAM per worker (fp32 weights
  with external-data sidecar)


## Evaluation

### Evaluation data

A single held-out evaluation sample of **600,000 positions** drawn
from Lichess rapid games played in **March 2026**, stratified at
100,000 positions per rating band. This sample is temporally held out
from Nova's training data and is shipped as `unified_sample_600k.pkl`
on Hugging Face.

### Metrics

- **hit1** — fraction of positions where the model's top prediction
  matches the human's actual move (top-1 accuracy)
- **hit5** — fraction of positions where the human's move is in the
  model's top-5 predictions
- **Mean P(actual)** — mean probability mass that the model assigned
  to the move the human actually played
- **Mean top-5 mass** — mean total probability mass assigned to the
  top-5 predictions

### Results

On the 600,000-position sample, comparing Nova against the publicly
available Maia-3 checkpoint (`maia3_simplified.onnx` from
https://maiachess.com) and the Maia-2 rapid checkpoint:

| Metric | Maia-2 | Maia-3 | Nova (neutral style) |
|---|---|---|---|
| Top-1 hit rate | 50.27 % | **54.83 %** | 54.60 % |
| Top-5 hit rate | 88.38 % | **91.23 %** | 91.10 % |
| Mean P(actual) | 38.44 % | 42.10 % | **42.51 %** |
| Mean top-5 mass | 89.33 % | 91.96 % | **92.26 %** |

All four Nova-vs-Maia-3 deltas are statistically significant under
paired McNemar tests (for hit rates) and paired t-tests (for
probability-mass metrics). Both probability-mass deltas remain
significant under a player-clustered bootstrap (95% CIs reported in
RESULTS.md).

Full breakdown by rating band, Maia tier (Skilled / Advanced /
Master), game phase, piece count, and three filter variants (all
positions; `ply ≥ 10`; `ply ≥ 10 + clock ≥ 30 s`) is in
[`RESULTS.md`](RESULTS.md).


## How to use

Minimum-dependency inference example (CPU):

```bash
pip install onnxruntime python-chess numpy
```

```python
import chess
import numpy as np
import onnxruntime as ort

PIECE = {"P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,
         "p":6,"n":7,"b":8,"r":9,"q":10,"k":11}

def fen_to_planes(fen):
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    parts = fen.split()
    board, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]
    for ri, rank_str in enumerate(board.split("/")):
        rank_idx, file_idx = 7 - ri, 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                planes[PIECE[ch], rank_idx, file_idx] = 1.0
                file_idx += 1
    if turn == "w":           planes[12].fill(1.0)
    if "K" in castling:       planes[13].fill(1.0)
    if "Q" in castling:       planes[14].fill(1.0)
    if "k" in castling:       planes[15].fill(1.0)
    if "q" in castling:       planes[16].fill(1.0)
    if ep != "-" and len(ep) == 2:
        planes[17, 0, ord(ep[0]) - ord("a")] = 1.0
    return planes

session = ort.InferenceSession("nova.onnx",
                               providers=["CPUExecutionProvider"])

board = chess.Board()
positions = fen_to_planes(board.fen())[np.newaxis]
# rating=1600, neutral classical + aggression
conditioning = np.array([[(1600 - 800) / (2700 - 800), 0.5, 0.5]],
                        dtype=np.float32)
logits = session.run(None, {"positions": positions,
                            "conditioning": conditioning})[0][0]

# Mask illegals + softmax
legal = np.zeros(16384, dtype=bool)
for mv in board.legal_moves:
    idx = mv.from_square * 64 + mv.to_square
    if mv.promotion == chess.KNIGHT:   idx += 4096
    elif mv.promotion == chess.BISHOP: idx += 4096 * 2
    elif mv.promotion == chess.ROOK:   idx += 4096 * 3
    legal[idx] = True
masked = np.where(legal, logits, -1e9)
probs = np.exp(masked - masked.max())
probs *= legal
probs /= probs.sum()

top = np.argsort(probs)[::-1][:5]
for i in top:
    print(f"  index {int(i):5d}  p = {probs[i]*100:.2f}%")
```

For production deployment notes (multi-worker setup, rate limiting,
temperature schedules, observability), see `docs/serving.md` in the
GitHub repository.


## Citation

```
Nova Chess Engine. Nova Chess, 2026.
https://github.com/novachessai/novachess-engine
https://huggingface.co/novachess/novachess-engine
```

BibTeX:

```bibtex
@misc{novachess_2026,
  title  = {Nova Chess Engine},
  author = {Nova Chess},
  year   = {2026},
  url    = {https://github.com/novachessai/novachess-engine}
}
```


## Acknowledgments

Nova builds on prior work in human-move prediction. The evaluation
methodology (rating-band stratification, tier definitions, ply-based
filters, Lichess rapid data) follows conventions established by the
Maia project.

- Maia-1 — McIlroy-Young, Sen, Kleinberg & Anderson, *Aligning
  Superhuman AI with Human Behavior: Chess as a Model System*,
  KDD 2020. [arXiv:2006.01855](https://arxiv.org/abs/2006.01855)
- Maia-2 — Tang, Jiao, McIlroy-Young, Kleinberg, Sen & Anderson,
  *Maia-2: A Unified Model for Human-AI Alignment in Chess*,
  NeurIPS 2024. [arXiv:2409.20553](https://arxiv.org/abs/2409.20553)
- Maia-3 — Maia project, https://maiachess.com. The specific
  checkpoint evaluated here is `maia3_simplified.onnx` published there.
- Allie — Khoshneviszadeh, Chi, Sheller et al., *Allie: Emergent
  Human-Like Play Through Adaptive MCTS with a Decoder-Only
  Transformer*, ICLR 2025.
  [arXiv:2410.03893](https://arxiv.org/abs/2410.03893)


## Contact

- Product website: <https://novachess.ai>
- GitHub issues: <https://github.com/novachessai/novachess-engine/issues>
- Commercial licensing inquiries: support@novachess.ai
