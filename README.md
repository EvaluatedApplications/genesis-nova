# Genesis-Nova — Technical Specification

> Status: working research prototype. This document describes the system **as built**, states the results
> that are **actually measured**, and is explicit about what has **not** been demonstrated. Nothing below is
> projected unless labelled "Hypothesis" or "Roadmap".

---

## 1. Summary

Genesis-Nova is a hybrid symbolic/neural runtime. Instead of training one end-to-end network to map inputs
to outputs, it trains a small recurrent controller (a GRU) to **operate a structured "platonic space"** — a
geometric substrate in which numbers, tokens, relations, and learned operations are first-class, inspectable
objects. The controller learns *which substrate operation to invoke*; the substrate performs it
deterministically.

Two properties fall out of that split and are why the architecture is worth attention:

1. **Exact computation, not approximation.** Arithmetic is computed by a closed-form homomorphism baked into
   the number representation, so it is exact and generalises to operands never seen in training — no per-fact
   memorisation.
2. **Inspectable behaviour.** Every answer carries a decision path (which route fired, which relation/face
   was used). It is not a black box; you can read why an output was produced.

It runs at small scale (512-dimensional space, small vocabularies, focused tasks). It is **not** a language
model and makes no language-model claims. See §5 for the limits.

---

## 2. Thesis (Hypothesis — not yet proven at scale)

End-to-end networks spend capacity re-learning things that are not statistical (exact arithmetic, identity,
composition) and store facts rather than structure. **Hypothesis:** a network that learns to *use* a
structured interface — compress into it, retrieve from it, compute with it — can be more data-efficient,
interpretable, and exact where it matters, because structure comes from the substrate rather than from data.
This document reports the mechanisms built to test that and the small-scale evidence so far. It does **not**
claim the hypothesis is validated (see §5–§7).

---

## 3. Architecture

### 3.1 The platonic space (representation)

A concept is a vector of dimension `D` (default 512) split into non-overlapping **faces**, each a functional
region (`Core/FaceLayout.cs`):

| Face | Range (D=512) | Holds | Encoding |
|---|---|---|---|
| Polynomial | `[0, 21)` | a number's value, **additively homomorphic** | `e[i] = value · 10^-(i+1)` |
| Logarithmic | `[21, 42)` | same value, **multiplicatively homomorphic** | `e[i] = ln\|value\| · 10^-(i+1)` |
| Character | `[42, 202)` | a token's spelling, one char per slot | per-char slot (generative) |
| Word | `[202, 512)` | whole tokens/phrases | per-word slot (recognition) |

The numeric faces are the key device. Because the encoding is linear in the value per dimension,
`poly(a) + poly(b) = poly(a+b)` and `log(a) + log(b) = log(a·b)` hold exactly. Arithmetic is therefore a
**vector operation plus a decode** — exact for any operands, with no stored facts
(`Core/PlatonicFaceComposer.cs`, `PlatonicFaceDecoder.cs`). A face is either *identity* (frozen — defines
what the concept is) or *free* (learnable), so e.g. the word "one" can migrate its free numeric face onto
the value 1 while its spelling stays fixed.

### 3.2 Concepts and relations as elements

Working memory (`Cognition/PlatonicSpaceMemory.cs`) holds two keyed indices of positioned elements:

- **Concepts** — points positioned by their faces; moved by a message-passing rule (pull related concepts
  together, push unrelated apart — contrastive positioning).
- **Relations** — each learned relation (`one ↔ 1`, `apple ↔ fruit`) is itself a **positioned element** at
  the centroid of its endpoints, with a strength (`1 − contradiction`). Relations are objects in the space,
  not a side-table; retrieval traverses them.

### 3.3 The neural controller

A GRU (`Model/GenesisNeuralModel.cs`, TorchSharp) with small task heads, lazily initialised, trained by
autograd (plus a REINFORCE-style head for the non-differentiable space edit):

- **Route head** (3-way): neural / platonic-direct / platonic-assisted.
- **Query heads**: classify the operation from context, and select which input tokens are operands.
- **Edit head**: how strongly to write to the space for a given input.

The controller never parses symbols. The operation for `3 x 2` vs `solve for x` is **classified from
context**, so a token is not locked to a single meaning.

### 3.4 Inference routing (`Infer/GenesisInferenceEngine.cs`)

The route head picks a path; each path is an inspectable resolver that *abstains* (falls through) when it
cannot answer confidently:

1. **GRU-query arithmetic** — op classified from context, computed by the face homomorphism.
2. **Learned-operation route** — apply a learned function/op (§3.5).
3. **Relation-first retrieval** — follow the strongest learned relation edge.
4. **Concept-chain** — bounded relational walk.
5. **Neural generation** (optionally with mid-generation platonic assist) — fallback.

### 3.5 Learned operations (few-shot, generalising)

An operation is learned from examples, stored as a substrate object, **selected from the space by relation**
(a cue concept → its related operation), and applied:

- **Unary** (`Core/TransformAccumulator.cs`): a function is a single translation vector
  `T(f) = avg(embed(out) − embed(in))`, learned with no gradient descent; applied by composition
  `embed(x) + T(f)` and decoded in the function's own face. Generalises affine/multiplicative functions to
  unseen operands from a handful of examples.
- **Binary** (`Core/FoldPathDiscovery.cs`): discovers structure — that an operation is a *fold* of a known
  one (× = repeated +) or fits a monomial law `c = a^α · b^β`.

This is bounded: it generalises operations expressible as constant translations / discovered folds (the
numeric/affine/structural class), **not** arbitrary symbolic maps, which the relation graph handles.

### 3.6 Training regime (`Train/CoreBootstrapRegime.cs`)

A focused curriculum trains one lesson to mastery before the next: mastery gating, learning-rate annealing
near target (to stop the ~90%-plateau oscillation), and rehearsal of mastered lessons to resist forgetting.
Convergence is *detected* (target held for a stability window), not run for a fixed budget.

---

## 4. Demonstrated results

All figures are produced by the automated test suite (109 tests, all passing) at the 512-dim configuration.
They are capability/stability demonstrations on focused tasks, **not** benchmark wins.

| Capability | Result | Meaning |
|---|---|---|
| Arithmetic (add/sub/mul/div) | Exact; generalises to unseen operands | Computed by the homomorphism, not memorised |
| Arithmetic op selection | Classified from context by the GRU | No symbol parser; `x` is math only in math context |
| Regime convergence (add / sub) | 86% / 91%, **held** to a stability window | Learned (stochastic) arithmetic, not parser-exact |
| Number-word equivalence (`one ↔ 1`) | Learned, bidirectional | Relation-as-element retrieval |
| Retention after new training | 36/36 prior probes retained (26/36 pre-fix) | Catastrophic forgetting measured and mitigated |
| Single-answer retrieval (`apple → fruit`) | Learned | Relational retrieval |
| Learned unary function (`+5`, `double`) | Generalises 4/4 to held-out operands from 4 examples | One-shot transform, no gradient descent |
| Learned binary op (`a·b` via fold discovery) | Generalises 3/3 to held-out operands | Structure discovered from examples |

**Engineering:** 109 automated tests; deterministic substrate (seeded); checkpoint save/load; decision-path
telemetry on every answer.

### 4.1 Head-to-head vs a transformer (equal budget)

A console harness (`bench/RaceBench`) races nova against a best-effort decoder-only transformer under a fair,
equal budget: **same tokenizer, same data, same epochs, matched parameter count.** The transformer is
competently configured (pre-LN blocks, multi-head attention, GELU, Adam, loss masked to the answer span —
all of which *help* it). Footprint, measured by the transparent formula `params × bytes` (nova SGD =
8 B/param: weights+grads; transformer Adam = 16 B/param: weights+grads+two moments):

> nova ≈ **1.69M params / ~13 MB** · transformer ≈ **1.64M params / ~25 MB** — **equal parameters, nova ~half the VRAM.**

**Full curriculum** (number-word equivalence, category retrieval, add/sub/mul/div), 20 flat epochs,
in-distribution held-out:

| | train | held-out |
|---|---|---|
| nova | 93% | **85%** |
| transformer | **99%** | 75% |

The transformer fits the training set *better* (99% vs 93%) but **generalises worse** (75% vs 85%) — nova
has the smaller train→held-out gap, at half the VRAM. On this in-distribution mix it is a near-match, not a
blow-out — that is the honest result.

**Arithmetic, extrapolation** (operands 21–40, never trained — a structural generalization test):

| | held-out (interp.) | extrapolation |
|---|---|---|
| nova | ~100% | **~99%** |
| transformer | ~9%* | ~3% |

Here the gap is decisive and *structural*: nova computes via the homomorphism so it extrapolates exactly;
the transformer interpolates statistics and fails outside the training range — a well-documented transformer
weakness that **more training does not fix**. (\*Under equal epochs the transformer is still under-trained on
interpolation; given many more epochs it would climb on in-range held-out — but **not** on extrapolation.)

**Reproduce:** `dotnet run --project bench/RaceBench/RaceBench.csproj -c Release`.

---

## 5. Limitations — what has NOT been shown

Deliberately blunt; this is the honest scope of the prototype.

- **Scale.** Everything above is at 512 dimensions, small vocabularies, short inputs. There is **no evidence
  the approach holds at large scale**, on real language, or on long-context reasoning.
- **Benchmark is small-scale and partial.** There is now a head-to-head (§4.1) vs an equal-budget transformer,
  but only at 512-dim on the focused curriculum. On in-distribution tasks it is a near-match (nova generalises
  a bit better at half the VRAM); the decisive, structural win is **arithmetic extrapolation**. This needs to
  be repeated at larger scale and on more tasks before the data-efficiency thesis is established.
- **Narrow tasks.** The demonstrated tasks (exact arithmetic, equivalence, category lookup, simple learned
  functions) validate *mechanisms*; they are not, by themselves, a product.
- **Thesis unproven.** That a learned interface beats end-to-end neural on a metric anyone cares about is a
  hypothesis with toy-scale support only (§2).
- **Generality bounds are real.** The learned-operation route generalises numeric/affine/structural
  operations, not arbitrary symbolic functions.
- **Prototype stack.** Single-process .NET / TorchSharp; not a hardened or distributed training system.

---

## 6. Engineering notes

- **Stack:** C# / .NET 8, TorchSharp (CUDA with CPU fallback), WPF desktop UI, REPL.
- **Codebase:** ~19k lines of source across ~100 files; 109 automated tests.
- **Reproducibility:** the substrate is deterministic under seed; results are regenerated by the test suite.
- **Discipline:** no hardcoded answer tables and no symbol parsers — capability is required to come from
  learned structure or the substrate's exact operations, and tests assert answers route through the platonic
  path rather than a memorising fallback.

---

## 7. Roadmap — what would validate the thesis

In priority order, because §5 names the gaps:

1. **A comparative benchmark.** Same focused tasks, a small transformer baseline, measured on
   data-efficiency / exactness / interpretability. This is what turns "interesting mechanisms" into
   "evidence the approach matters."
2. **Scale-up study.** Larger space / vocabulary / input length; measure whether the properties (exactness,
   retention, few-shot operations) survive.
3. **Controller-composed operations.** Let the GRU compose substrate operations itself (the composable block
   vocabulary exists and is tested) rather than via fixed routes — open-vocabulary capability.
4. **Unify learned operations into the space** as first-class elements (relations already made this move).

---

## Build, test, run

```bash
dotnet build GenesisNova.slnx
dotnet test  GenesisNova.slnx
```

REPL:

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-repl
```

Useful REPL commands: `train <input> => <output>`, `trainfile <path> [epochs]`, `context`, `reset`.
The desktop UI (`src/GenesisNova/UI/`) exposes training controls; checkpoints persist to disk. Training
corpora live in `data/` (main: `data\genesis-nova-train-expanded.txt`).
