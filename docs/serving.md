# Serving Nova

Practical notes for running Nova in production — loading the model,
masking illegal moves, choosing a sampling temperature, and
operational tips for running it as an HTTP service.

The README's quickstart shows the minimum code needed to get a
single prediction. This doc adds what's useful beyond that.


## Loading the ONNX model

Nova is distributed as two files:

- `nova_v3b.onnx` — the graph (≈ 1 MB)
- `nova_v3b.onnx.data` — the weights, external-data sidecar (≈ 400 MB)

Both files must sit in the same directory when loading **and keep
their exact filenames** — the `.onnx` graph file embeds a reference
to `nova_v3b.onnx.data` by name. Renaming the sidecar will break loading.
You only pass the `.onnx` path to `InferenceSession`; the runtime
resolves the sidecar automatically:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "nova_v3b.onnx",
    providers=["CPUExecutionProvider"],  # or CUDAExecutionProvider
)
```

If you move one file without the other, the session will either fail
to load or load with uninitialized weights (depending on runtime
version). Keep them together.

Model outputs are identified by name: the only output is `logits`,
of shape `(B, 16384)`. Inputs are `positions` (shape `(B, 18, 8, 8)`)
and `conditioning` (shape `(B, 3)`).


## Legal-move masking

Nova's raw output is logits over all 16,384 possible `(from, to,
promotion)` combinations, including illegal ones. Mask before
softmax — otherwise you'll sometimes sample moves the position
doesn't support. Reference pattern:

```python
import chess
import numpy as np

def build_legal_mask(board):
    """Return a (16384,) boolean array marking legal move indices."""
    mask = np.zeros(16384, dtype=bool)
    for mv in board.legal_moves:
        idx = mv.from_square * 64 + mv.to_square
        if mv.promotion == chess.KNIGHT:   idx += 4096
        elif mv.promotion == chess.BISHOP: idx += 4096 * 2
        elif mv.promotion == chess.ROOK:   idx += 4096 * 3
        # queen promotion keeps the default offset 0
        mask[idx] = True
    return mask

def masked_softmax(logits, mask):
    """Softmax over legal indices only, returning a (16384,) probability vector."""
    masked = np.where(mask, logits, -1e9)
    m = masked.max()
    probs = np.exp(masked - m)
    probs *= mask           # zero out illegal slots exactly
    probs /= probs.sum()
    return probs
```

Apply this to every position before sampling or reading top-K. The
model is not guaranteed to keep illegal-move logits bounded (they
can be arbitrarily large), which is why we mask *before* softmax
rather than setting probabilities to zero afterward.


## Temperature sampling

Nova returns a probability distribution; you can play it as argmax
(deterministic) or sample with a temperature. Two principles cover
most use cases:

- **Lower temperature → stronger, more repetitive play.** As
  temperature approaches 0, sampling collapses toward the top-1
  argmax. Use this when you want Nova's strongest play (e.g.,
  benchmarking, high-rating bots, sharp endgames).
- **Higher temperature → more variety, weaker play.** Sampling
  approaches the raw policy distribution. Use this when you want
  opening variety, lower-rated play, or a more diverse pool of
  candidates.

Sampling implementation:

```python
def sample_with_temperature(probs, temperature):
    if temperature <= 0.01:
        return int(np.argmax(probs))
    # Re-temper: raise unnormalized probs to 1/T
    p = probs ** (1.0 / temperature)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```

Beyond a single fixed temperature, two patterns are commonly useful:

- **Opening variety.** Sampling at higher temperature for the first
  few plies (then dropping for the rest of the game) prevents Nova
  from always playing the same first move while preserving stronger
  middlegame/endgame play.
- **Endgame sharpness.** Lowering temperature in deep endgames
  (e.g., piece count ≤ 5–8) helps Nova convert technical wins and
  hold drawable positions, since endgame play is more tactically
  rigid than middlegame.

Calibrating Nova to a specific target playing strength is mostly a
matter of finding the right temperature schedule and combining it
with an external evaluator (e.g., Stockfish or any postion-valuation model) for blunder filtering
when a precise rating target matters. Implementations of that stack
are out of scope for this document — the released checkpoint is the
policy network only.


## Running as an HTTP service

The README's inference example is synchronous and single-threaded.
For a production service you'll likely want:

- **A persistent FastAPI/Flask process** that loads the model once
  at startup and reuses the session across requests. ONNX load time
  is ~0.5 s and dominant cost; don't pay it per request.
- **Multiple uvicorn workers** if you're serving concurrent users.
  Each worker loads its own copy of the model (~500 MB RAM per
  worker) and processes requests independently. Two workers give
  roughly 2× throughput up to CPU saturation.
- **A legal-move mask built per request** (the mask depends on the
  position, so it can't be cached globally).
- **Optional: return top-K candidates with probabilities** instead
  of a single sampled move. This lets the client choose sampling
  strategy, and is more useful for UIs that want to display
  move-quality heatmaps.


## Rate limiting + access control

If you're exposing Nova as an open HTTP endpoint, rate-limit on the
edge:

- **Per-IP rate limit** (e.g., 60 req/min) handles casual abuse
  without requiring authentication. Standard middleware exists for
  Caddy (`caddy-ratelimit`), Nginx (`limit_req`), Cloudflare
  (per-zone rules), and most reverse proxies.
- **Shared-secret API key** in a custom header (`X-API-Key`) is the
  simplest authenticated-access model. Adequate for embedding the
  model behind a frontend you control.
- **Per-user tokens** tied to accounts — if your deployment has
  login/auth already, reuse that. Only worth the engineering
  overhead once you have concrete commercial / per-user-quota
  requirements.

Inference itself is CPU-bound and won't ddos easily — the main
resource to protect is bandwidth and the principal bottleneck is
concurrent CPU threads.


## Logging + observability

One persistent access log per process is usually sufficient. For
each request, log: timestamp, client IP, latency, response code,
and the `(rating, style)` conditioning the client asked for.
That's enough to (a) detect abuse, (b) understand usage patterns
at each rating band, and (c) debug latency regressions.

Python/uvicorn default access logs cover most of this; pipe stderr
to a rotated log file (`logrotate` on Linux).


## Performance notes

| Configuration | Typical latency | Notes |
|---|---|---|
| CPU, ONNX fp32, single thread | 35–50 ms / position | single position at a time; dominated by matmul |
| CPU, ONNX fp32, 2 uvicorn workers | same per worker | ~40–50 req/s aggregate throughput on a modest VPS |
| GPU (H100), batched inference | ~1 ms / position | if you batch, throughput scales to 1000+ pos/s |

Future releases may include alternative model architectures, different
parameter counts, or additional export formats (including quantized
variants) optimized for different deployment targets. Nothing
specific is committed; check the Hugging Face repository for the
current set of available artifacts.
