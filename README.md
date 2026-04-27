# Nova Chess Engine

Nova is a style-conditioned transformer that predicts human chess
moves. Given a position, a target rating, and two style parameters,
it returns a probability distribution over all legal moves calibrated
to how a player of that rating and style would play.

**A single forward pass per position — no tree search, no MCTS, no
minimax, no position evaluation, no value head.** Every move Nova
produces comes from one pass through the neural network. This is
fundamentally different from search-based engines like Stockfish
(alpha-beta over NNUE evaluations) or Leela (MCTS guided by a
policy + value head), and it's also more spartan than other
human-move predictors: Maia-2 ships a policy head **plus** a value
head (game-outcome regression) **plus** an auxiliary head (legal
moves, captures, check delivery); Maia-3 keeps the value head;
Allie wraps its policy in adaptive MCTS at inference time. Nova
ships only the policy. The choice is deliberate — humans don't
run search trees either, and dropping the value head forces the
network to learn move quality entirely through learned move-
selection patterns rather than offloading it to a parallel
evaluator. Inference cost: ~35–50 ms per position on a single
CPU core.

Nova is the engine behind **[novachess.ai](https://novachess.ai)** —
the chess-learning platform where you can play or practice against
Nova directly at chosen rating and style settings.

- Conditions on `rating ∈ [800, 2700]`, `classical ∈ [0, 1]`, `aggression ∈ [0, 1]`
- Trained on Lichess rapid games, balanced across rating bands 800–2700+
- Published as ONNX for CPU inference (~35–50 ms / position)
- **Single-head, pure-policy architecture** — no value head, no auxiliary head, no search, no lookahead. The only head is the move-distribution policy. The only thing the network ever predicts is the next move.

**Links**
- Product: <https://novachess.ai>
- Model weights + evaluation artifacts: <https://huggingface.co/novachess/novachess-engine>


## Benchmark headline

Evaluated on **600,000 positions** from Lichess rapid games played in
March 2026 (temporally held out from Nova's training set), against
the publicly available Maia-3 checkpoint (`maia3_simplified.onnx`)
and the Maia-2 `rapid_model.pt` baseline. Nova is evaluated in
neutral style (`classical = 0.5, aggression = 0.5`) for apples-to-apples
comparison against Maia, which has no style conditioning.

| Metric | Maia-2 | Maia-3 | **Nova** | Nova vs Maia-3 |
|---|---|---|---|---|
| Top-1 hit rate | 50.27 % | **54.83 %** | 54.60 % | Maia-3 by 0.23 pp |
| Top-5 hit rate | 88.38 % | **91.23 %** | 91.10 % | Maia-3 by 0.13 pp |
| Mean P(actual move) | 38.44 % | 42.10 % | **42.51 %** | Nova by 0.41 pp |
| Mean top-5 probability mass | 89.33 % | 91.96 % | **92.26 %** | Nova by 0.30 pp |

Maia-3 is ahead on argmax agreement (top-1, top-5). Nova is ahead on
probability mass (assigning density to the actual move, and to the
top-5 set). All four Nova-vs-Maia-3 deltas are statistically
significant under paired McNemar / paired t-tests and remain
significant under a player-clustered bootstrap.

Full breakdown by rating band, by Maia tier, by game phase, and by
piece count is in **[RESULTS.md](RESULTS.md)** — along with delta
tables and sub-filter variants (ply ≥ 10; ply ≥ 10 + clk ≥ 30 s).


## How Nova compares to other chess models

| Model | Heads at inference | Search | Notes |
|---|---|---|---|
| **Nova** (this release) | 1 (policy) | none | single forward pass, ~35–50 ms CPU |
| Maia-2 (NeurIPS 2024) | 3 (policy + value + auxiliary) | none | value head regresses W/D/L; auxiliary head predicts legal moves, captures, check delivery |
| Maia-3 (`maia3_simplified.onnx`) | 2 (policy + value) | none | drops Maia-2's auxiliary head; retains W/D/L value head |
| Allie (ICLR 2025) | 1 (policy) + value via search | adaptive MCTS at inference | policy is decoder-only over move sequences; MCTS provides per-position evaluation at runtime |
| Leela (LC0) | 2 (policy + value) | MCTS | engine-strength playing model |
| Stockfish (NNUE) | evaluation only | alpha-beta | not a human-move predictor |

Nova ships only the move-policy head — no value head, no auxiliary
head, no MCTS, no alpha-beta, no lookahead of any kind. The
benchmarks above show this is competitive with multi-head architectures
on the move-prediction task itself.


## Installation

```bash
pip install onnxruntime python-chess numpy
```


## Quick start — run inference on a position

Download the ONNX files from Hugging Face:

```bash
huggingface-cli download novachess/novachess-engine \
  nova.onnx nova.onnx.data --local-dir .
```

Or clone the full HF repo. Both `nova.onnx` and
`nova.onnx.data` are required — the `.data` file holds the
weights (external-data format).

Minimal inference example:

```python
import chess
import numpy as np
import onnxruntime as ort

# --- board encoding (matches training) ---
PIECE = {"P":0,"N":1,"B":2,"R":3,"Q":4,"K":5,
         "p":6,"n":7,"b":8,"r":9,"q":10,"k":11}

def fen_to_planes(fen: str) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    parts = fen.split()
    board_part, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]
    for ri, rank_str in enumerate(board_part.split("/")):
        rank_idx = 7 - ri
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                planes[PIECE[ch], rank_idx, file_idx] = 1.0
                file_idx += 1
    if turn == "w":                     planes[12].fill(1.0)
    if "K" in castling:                 planes[13].fill(1.0)
    if "Q" in castling:                 planes[14].fill(1.0)
    if "k" in castling:                 planes[15].fill(1.0)
    if "q" in castling:                 planes[16].fill(1.0)
    if ep != "-" and len(ep) == 2:
        planes[17, 0, ord(ep[0]) - ord("a")] = 1.0
    return planes

def normalize_rating(r: int) -> float:
    r = max(800, min(2700, r))
    return (r - 800) / (2700 - 800)

def decode_move_idx(idx: int) -> str:
    promo = ""
    raw = int(idx)
    if raw >= 4096 * 3:   promo = "r"; raw -= 4096 * 3
    elif raw >= 4096 * 2: promo = "b"; raw -= 4096 * 2
    elif raw >= 4096:     promo = "n"; raw -= 4096
    return chess.square_name(raw // 64) + chess.square_name(raw % 64) + promo

# --- load model ---
session = ort.InferenceSession("nova.onnx",
                               providers=["CPUExecutionProvider"])

# --- run on a position ---
board = chess.Board()  # starting position
positions = fen_to_planes(board.fen())[np.newaxis]  # (1, 18, 8, 8)
conditioning = np.array([[normalize_rating(1600), 0.5, 0.5]],
                        dtype=np.float32)

logits = session.run(None, {"positions": positions,
                            "conditioning": conditioning})[0][0]

# mask illegal moves
legal = np.zeros(16384, dtype=bool)
for mv in board.legal_moves:
    idx = mv.from_square * 64 + mv.to_square
    if mv.promotion == chess.KNIGHT:  idx += 4096
    elif mv.promotion == chess.BISHOP: idx += 4096 * 2
    elif mv.promotion == chess.ROOK:   idx += 4096 * 3
    legal[idx] = True

masked = np.where(legal, logits, -1e9)
probs = np.exp(masked - masked.max())
probs *= legal
probs /= probs.sum()

# top 5 predictions
top = np.argsort(probs)[::-1][:5]
for i in top:
    print(f"  {decode_move_idx(int(i)):6s}  p = {probs[i]*100:5.2f}%")
```

See `docs/serving.md` for production deployment notes — legal-move
masking patterns, temperature scheduling, multi-worker setup, rate
limiting, and observability.


## Conditioning inputs

Nova takes three scalar conditioning values alongside each position:

| Input | Range | Effect |
|---|---|---|
| `rating` | 800 – 2700 | Player strength. Normalized linearly to [0, 1] internally. |
| `classical` | 0 – 1 | Opening preference. Higher = more classical (1.e4, 1.d4 mainlines). Lower = more hypermodern (1.Nf3, 1.c4, 1.b3, 1.g3, fianchetto). Strongest effect in the opening; milder in the middlegame. |
| `aggression` | 0 – 1 | Tactical / sacrificial tendency. Higher = more forcing moves, captures, king attacks. Conditioned throughout the game. |

For a generic "human of rating R" comparison, use `classical = 0.5`
and `aggression = 0.5`. This is the "neutral style" configuration
used for all Maia comparisons in [RESULTS.md](RESULTS.md).


## Move representation

The policy head outputs a distribution over 16,384 move indices:

```
move_index = promotion_offset + from_square * 64 + to_square

promotion_offset:
    0      no promotion  (also queen promotion)
    4096   knight promotion
    8192   bishop promotion
    12288  rook promotion
```

`from_square` and `to_square` are 0–63 in standard chess ordering
(a1 = 0, h8 = 63).


## Model I/O

- **Input `positions`**: `float32` tensor `(B, 18, 8, 8)` — 18-plane
  encoding of the board. See the quickstart above for the encoder.
  - Planes 0–5: white pieces (P, N, B, R, Q, K), one-hot per square
  - Planes 6–11: black pieces
  - Plane 12: side to move
  - Planes 13–16: castling rights (KQkq)
  - Plane 17: en-passant file
- **Input `conditioning`**: `float32` tensor `(B, 3)` — `(rating_norm,
  classical, aggression)` with `rating_norm = (rating − 800) / 1900`.
- **Output `logits`**: `float32` tensor `(B, 16384)` — raw logits over
  the 16,384-index move space. Callers must apply a legal-move mask
  before sampling; the quickstart above shows the standard pattern.

Internal architecture details (layer counts, dimensions, attention
configuration, conditioning injection mechanism, etc.) are not
published.


## Performance

| Hardware | Latency | Throughput |
|---|---|---|
| Modern x86 CPU core (ONNX fp32) | 35–50 ms / position | ~20–25 req/s single-threaded |
| Multi-worker CPU server | same per worker | scales ~linearly with workers |
| H100 GPU (batched) | ~1 ms / position | > 1000 positions/s |

Memory: ~500 MB RAM per inference worker (ONNX fp32, external data).


## Repository layout

```
novachess-engine/
├── README.md              this file
├── MODEL_CARD.md          HF-style model card (mirrored on Hugging Face)
├── RESULTS.md             evaluation breakdown on the 600K sample
├── LICENSE                usage terms (see LICENSE)
└── docs/
    └── serving.md         inference and deployment notes
```

Weights and the 600K evaluation sample are hosted on Hugging Face:

- `nova.onnx` + `nova.onnx.data` — model graph + weights
- `unified_sample_600k.pkl` — the validation sample used in RESULTS.md

<https://huggingface.co/novachess/novachess-engine>


## License

Custom non-commercial license. In short:

- Use, modify, run, fine-tune, benchmark for personal / research /
  educational purposes: **allowed**
- Operate a free public bot (Lichess, Discord, etc.) that uses Nova
  with attribution: **allowed**, including with donations / tips
- Sell, license, or include Nova in a paid product or service:
  **requires a commercial license** — contact support@novachess.ai
- Fine-tunes and distilled derivatives inherit this license
- Model outputs (games, PGNs, evaluations) are unencumbered

Full text: [LICENSE](LICENSE). Attribution template:

> Powered by Nova Chess — https://github.com/novachessai/novachess-engine


## Try Nova at novachess.ai

The easiest way to experience the model is at
**[novachess.ai](https://novachess.ai)** — play full games or
drill-style practice positions against Nova at any rating and style
setting, with built-in analysis, hints, and post-game review. The
version of Nova served by the app uses the same weights published in
this repository.


## In-app behavior vs the released model

The OSS release is **Nova — the pure policy network**, exactly as
described in this README and `MODEL_CARD.md`. One forward pass per
position; no search, no value head, no post-processing.

The version served by the [novachess.ai](https://novachess.ai) *Play*
page is calibrated across rating tiers from approximately **chess.com
500 to 2500 blitz**, using two lightweight calibration layers on top
of the same policy weights published here:

- **Per-tier temperature scheduling.** The primary calibration lever.
  Higher temperature flattens the policy distribution (more variety,
  more lower-rated mistakes); lower temperature concentrates it
  (closer to argmax, cleaner play). Each rating tier is mapped to its
  own temperature schedule so the bot's per-phase mistake profile
  matches the chess.com CP-loss profile at that level.
- **Evaluation-only filter.** At higher tiers, after Nova samples a
  candidate move, Stockfish is consulted at low depth to evaluate
  whether that specific candidate falls below a tier-dependent quality
  threshold. If it does, the move is probabilistically replaced by
  re-sampling from Nova's own distribution (with the rejected move
  removed). The same evaluator filters voluntary draw offers when the
  position is winning.

**Every move the in-app bot plays still originates from Nova's policy
distribution.** Stockfish is never used to suggest, generate, or
select moves — only to evaluate moves Nova has already proposed, so
that obvious blunders at higher tiers can be probabilistically caught.
The model weights are never touched. The calibration layer is **not**
part of this release.

The OSS release stays pure-policy for three reasons:

1. **Clean benchmarking surface.** Maia, Allie, and other human-move
   predictors are also released as bare policy models; the right
   apples-to-apples comparison is policy vs policy.
2. **Most downstream uses don't want it.** Research, fine-tuning,
   move-humanness scoring, and anti-cheat all want the unmodified
   probability distribution Nova learned, not a downstream filter on it.
3. **External dependencies.** The calibration layer requires Stockfish
   at serving time — not appropriate to require of OSS users who just
   want to load the weights.

If you want to play the per-tier-calibrated in-app experience, visit
[novachess.ai](https://novachess.ai). If you want the raw model for
your own research, training, or downstream system, use the weights in
this repository directly — they are exactly the policy network
described here.


## Citing Nova

If you use Nova in research or write about it publicly, please cite:

```
Nova Chess Engine. Nova Chess, 2026.
https://github.com/novachessai/novachess-engine
```


## Acknowledgments

Nova builds on prior work in human-move prediction. The evaluation
methodology in this repository — rating-band stratification, tier
definitions, ply-based filters, Lichess rapid data — follows
conventions established by the Maia project.

- **Maia-1** — McIlroy-Young, Sen, Kleinberg & Anderson, *Aligning
  Superhuman AI with Human Behavior: Chess as a Model System*,
  KDD 2020. [arXiv:2006.01855](https://arxiv.org/abs/2006.01855)
- **Maia-2** — Tang, Jiao, McIlroy-Young, Kleinberg, Sen & Anderson,
  *Maia-2: A Unified Model for Human-AI Alignment in Chess*,
  NeurIPS 2024. [arXiv:2409.20553](https://arxiv.org/abs/2409.20553)
- **Maia-3** — Maia project, <https://maiachess.com>. The specific
  checkpoint evaluated in this repository is `maia3_simplified.onnx`
  published there.
- **Allie** — Khoshneviszadeh, Chi, Sheller et al., *Allie: Emergent
  Human-Like Play Through Adaptive MCTS with a Decoder-Only
  Transformer*, ICLR 2025. [arXiv:2410.03893](https://arxiv.org/abs/2410.03893)
