# microgpt-go

The most atomic way to train and run inference for a GPT — in pure, dependency-free Go.

Ported from [@karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) and reimagined with tape-based autograd, fused ops, and tape-free inference. Single file, zero dependencies, full Unicode support.

## Quick start

```bash
go build -o microgpt .
./microgpt
```

This trains a character-level GPT on [`input.txt`](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt) (32K English names) and generates new ones:

```
num docs: 32032
vocab size: 27
num params: 4192
step 5000 / 5000 | loss 1.8798 | avg100 1.5108
--- inference ---
aaliyah
arianna
kaylani
...
```

## How it works

The entire model — autograd, tokenizer, transformer, optimizer, checkpoint, inference — lives in a single file [`microgpt.go`](microgpt.go). No frameworks, no BLAS, no cgo.

**Architecture** (same as microgpt.py, compatible weights):
- GPT-2 style transformer with RMSNorm (no biases, ReLU instead of GeLU)
- Character-level tokenizer with BOS token
- Multi-head causal attention with KV-cache
- AdamW optimizer with decoupled weight decay, linear warmup + decay
- `Config` struct holds all hyperparameters — no global mutable state

**Key differences from the Python original:**

| | microgpt.py | microgpt-go |
|---|---|---|
| Autograd | DAG + topological sort | Tape (Wengert list), O(1) node creation |
| Node size | ~200 bytes (Python object) | 32 bytes (compact struct) |
| Sub/Div/Neg | Compound (3/2/2 nodes) | Native primitives (1 node each) |
| Dot product | 2N-1 nodes per call | Fused `opDot` (1 node) |
| Pow exponents | Python dict | Dense slice (indexed via node field) |
| Inference | Full autograd overhead | Tape-free pure float64 with workspace reuse |
| Tape reuse | New graph every step | `Reset()` keeps backing arrays |
| Allocations | Per-call slices | Pre-allocated workspace structs + KV pool |
| Unicode | Python str (native) | Rune-based tokenizer (full UTF-8) |
| Checkpoints | None | Binary save/load with resume + validation |

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `-data` | `input.txt` | Path to dataset (one document per line) |
| `-layers` | `1` | Number of transformer layers |
| `-embd` | `16` | Embedding dimension |
| `-ctx` | `16` | Context window size |
| `-heads` | `4` | Number of attention heads |
| `-steps` | `5000` | Number of training steps |
| `-lr` | `0.01` | Learning rate |
| `-temp` | `0.5` | Sampling temperature |
| `-samples` | `20` | Number of generated samples |
| `-seed` | `0` | Random seed (0 = random) |
| `-batch` | `1` | Batch size for data-parallel training |
| `-wd` | `0` | Weight decay (AdamW decoupled L2, 0 = disabled) |
| `-save` | | Save checkpoint after training |
| `-load` | | Load checkpoint before training |

## Examples

**Train on English names (default):**

```bash
./microgpt -steps 5000
```

**Train a larger model on a custom dataset (with weight decay):**

```bash
./microgpt \
  -data words.txt \
  -embd 128 -heads 4 -layers 4 -ctx 48 \
  -steps 2000000 -lr 0.003 -wd 0.01 \
  -save model.bin
```

**Train with data-parallel batching (4 documents per step):**

```bash
./microgpt \
  -data words.txt \
  -embd 128 -heads 4 -layers 4 -ctx 48 \
  -steps 500000 -lr 0.003 -batch 4 \
  -save model.bin
```

**Resume training from a checkpoint:**

```bash
./microgpt -data words.txt -load model.bin -steps 500000 -save model2.bin
```

LR warmup (5% of steps) prevents gradient explosion on resume — no manual LR tuning needed.

**Inference only (no training):**

```bash
./microgpt -load model.bin -steps 0 -temp 0.3 -samples 50
```

## Checkpoint format

Binary, little-endian, version 2:

```
[8]byte   magic        "MGPT\0\0\0\0"
uint32    version      2
int32×4   hyperparams  [nLayer, nEmbd, blockSize, nHead]
int32     numRunes     number of unique characters
int32     numBytes     UTF-8 byte length of character set
[]byte    chars        UTF-8 encoded characters
int32     step         training step (for Adam bias correction)
int32     numParams    total parameter count
[]float64 data         model weights
[]float64 m            Adam first moment
[]float64 v            Adam second moment
```

Hyperparameters, tokenizer, and optimizer state are all restored automatically on load. Checkpoint loading validates sizes (1 GB allocation limit) and checks `nEmbd % nHead == 0`.

## Training features

- **LR schedule**: `computeLR()` — linear warmup (5%) → linear decay to zero (safe for `totalSteps=0`)
- **Gradient clipping**: global L2 norm capped at 1.0
- **Weight decay** (`-wd`): AdamW decoupled L2 regularization — `θ *= (1 - lr × wd)` before Adam step. Default 0 (disabled), recommended 0.01–0.1 for larger models
- **Bad batch recovery**: NaN/Inf gradients skip Adam update (params preserved); 10 consecutive → early stop
- **Epoch shuffle**: dataset reshuffled each epoch, every sample seen exactly once per epoch
- **Circular loss buffer**: fixed-size ring buffer for running average (constant memory regardless of step count)
- **Data-parallel batching** (`-batch N`): processes N documents in parallel using separate tapes per worker, averages gradients, and performs a single Adam update per step. With `-batch 1` (default), training is identical to the original sequential path. Gradient aggregation and Adam updates are also parallelised across CPU cores for large models (≥4K params).

## Performance

Benchmarks on the default config (1 layer, embd=16, ctx=16, vocab=27, Apple M1 Pro):

| | Training step (batch=1) | Training step (batch=4) | Inference (generate) |
|---|---|---|---|
| Latency | ~158 μs | ~280 μs (4 docs) | ~32 μs |
| Per-document | ~158 μs | ~70 μs (**2.25x**) | — |
| Memory | ~25 KB | ~131 KB | ~10 KB |
| Allocs | ~87 | ~328 | ~46 |

> **Note on batch mode:** Each step with `-batch N` processes N documents, so to see the same amount of data you need proportionally fewer steps. For example, `50000` steps at batch=1 sees 50K documents; the equivalent is `6250` steps at batch=8 — but ~3.6x faster in wall time. The speedup grows with model size; for larger models (embd=128+, layers=4+) expect near-linear scaling with CPU cores as forward+backward dominates over goroutine overhead.

Key optimizations:
- **Workspace structs** (`fwdWorkspace`, `inferWorkspace`): pre-allocated buffers reused across positions/samples, eliminating per-call allocations in forward passes
- **KV cache pooling**: `kvCache` and `inferKV` pre-allocate a flat buffer for all KV entries, eliminating `2 × nLayer × nPositions` allocations per step
- **`Into` function variants** (`linearInto`, `rmsnormInto`, `softmaxInto`, `softmaxF64Into`, etc.): write into provided buffers instead of allocating
- **Cached tape constants**: rmsnorm's `1/n` and `ε` leaves are created once per forward call and reused across all layers/positions
- **Dense `powExps` slice**: replaces `map[Idx]float64` for pow exponents, eliminating map lookups in backward pass hot path
- **Tape reuse**: `Reset()` clears nodes but keeps backing arrays across training steps

## Tests

```bash
go test ./... -count=1
go test ./... -count=1 -race
go test -bench=. -benchmem
```

Test coverage:
- Numerical gradient verification for all tape ops (Add, Mul, Pow, Log, Exp, ReLU, Neg, Sub, Div, Dot, Sum)
- Deterministic training (same seed → identical losses, both single and batch modes)
- Tape reuse correctness
- Inference-vs-training forward pass consistency
- Checkpoint round-trip (save/load/resume)
- Checkpoint validation (bad magic, huge allocations, invalid hyperparams)
- Tokenizer edge cases (single-char documents, encode/decode roundtrip)
- Long line handling in data loading (100KB+ lines)
- LR schedule correctness (warmup/decay boundaries, `totalSteps=0` edge case)
- Weight decay reduces parameter norm vs no decay
- Generate completes without panic on fresh params
- Batch training: single-equivalence, determinism, quality (loss decreases)

## License

MIT License. See [LICENSE](LICENSE).
