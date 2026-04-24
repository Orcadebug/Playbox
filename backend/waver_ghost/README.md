# Waver Ghost

Optional edge-side short-circuit service for chunked uploads.

## Purpose

Clients that opt into chunked upload pay nothing for obvious zero-hit queries: the
ghost proxy keeps tiny probabilistic summaries of what each payload has seen and
rejects misses before the main retrieval stack is touched. Non-chunked clients bypass
it entirely.

## What's inside `GhostProxy`

Two real, fixed-memory sketches (previous versions stored full SHA-256 hex digests in
`set[str]` / `dict[str, int]`, which grew unbounded and OOM'd on large ingests — that
is no longer the case):

- **Bloom filter** — `bytearray` bit-array of `m` bits with `k` hash functions via
  Kirsch–Mitzenmacher double-hashing over a single `blake2b(digest_size=16)` digest
  (two 64-bit halves). Defaults: `m = 1 << 20` bits (128 KiB), `k = 7` — ~1% FPR at
  ~100k items. Power-of-two sizing so `& mask` replaces `%`.
- **Count-Min Sketch** — `array.array("I", …)` shaped `depth × width` with explicit
  `< 0xFFFFFFFF` saturation guard (stdlib `array("I")` raises `OverflowError` at
  `2**32 - 1`, it does not auto-saturate). Defaults: `depth = 4`, `width = 1 << 17`
  (~2 MiB). `query()` returns the min across rows.

Memory is **O(1) in payload size**: ~128 KiB + ~2 MiB regardless of how much data
streams through. `payload_id` is folded into the hashed key so multiple payloads share
one global sketch.

## Saturation rotation

Because the sketch is global, ingesting across many payloads will eventually push FPR
toward 1.0 and make the short-circuit useless. `GhostProxy` tracks `bits_set`
incrementally and estimates load via `approx_items()` using the standard Bloom
cardinality formula `-m/k · ln(1 − bits_set/m)`. When that estimate crosses
`saturation_threshold` (default 150 000, ≈ 1.5× design capacity), it calls
`reset()` — zeroes both sketches and increments `rotations`. Callers can also invoke
`reset()` explicitly to rotate between workspaces or cold-path refreshes.

## Public API

```python
from waver_ghost import GhostProxy, GhostVerdict

proxy = GhostProxy()  # optional: bloom_bits=, bloom_hashes=, cms_width=, cms_depth=,
                      #           saturation_threshold=
proxy.ingest("payload-1", ["customer billing refund issue", "shipping update"])
verdict: GhostVerdict = proxy.query("payload-1", "billing refund")
# verdict.maybe_hit, verdict.bloom_overlap, verdict.cms_score

proxy.approx_items()   # estimated unique insertions
proxy.rotations        # how many auto-resets have fired
proxy.reset()          # zero both sketches, reset bits_set
```

Non-zero `maybe_hit` is probabilistic (false positives possible, false negatives
impossible). `cms_score` is a CMS min-estimate of term frequency.

## Integration status

`GhostProxy` is a library module — it is **not** wired into the request path yet.
Callers that want to use it should hold an instance per workspace / cold-path tier and
call `ingest()` as payloads stream in, `query()` before routing into the full
retrieval stack. See `backend/tests/test_ghost.py` for hit/miss, memory-bound, FPR,
overflow, and rotation coverage.
