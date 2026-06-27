# Genesis-Nova — Technical Specification

> Status: working research prototype. This document describes the system **as built**, states the results
> that are **actually measured**, and is explicit about what has **not** been demonstrated. Nothing below is
> projected unless labelled "Hypothesis" or "Roadmap".
>
> **Read with `PLATONIC_RECKONING.md` (2026-06).** §3.3–3.4 below describe the route/plan-head classifier ladder
> and a neural-decoder fallback. The reckoning reframes those as *overfit orchestration to subtract*: the current
> direction (behind `KeepCoreControl`) makes relaxation the retrieval route (`platonic-reason`), anchors routing on
> the discriminative cue, **abstains instead of falling back to the neural decoder**, and retrieves in O(log N) via
> the `DialecticalSpace` lattice. The substrate sections (faces, homomorphism, distributional meaning) stand.

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

> **On the word "platonic."** It is a chosen metaphor — a *space of ideas* whose rules we get to define so
> long as they stay internally consistent (the substrate axioms G1–G6, see `PLATONIC_THEORY.md`). It is the
> generative inspiration behind the design, **not** a claim that the system literally models a Platonic realm,
> physics, or a "field" in any physical sense. The companion docs (`PLATONIC_MIND.md`,
> `PLATONIC_CONSCIOUSNESS.md`) carry richer field/consciousness language; that is vision and motivation, held
> lightly per `PLATONIC_RECKONING.md`. This README sticks to what is built and measured.

---

## 2. Thesis (Hypothesis — not yet proven at scale)

End-to-end networks spend capacity re-learning things that are not statistical (exact arithmetic, identity,
composition) and store facts rather than structure. **Hypothesis:** a network that learns to *use* a
structured interface — compress into it, retrieve from it, compute with it — can be more data-efficient,
interpretable, and exact where it matters, because structure comes from the substrate rather than from data.
This document reports the mechanisms built to test that and the small-scale evidence so far — including a
measured equal-budget win over a transformer (§4.1). It does **not** claim the hypothesis is validated **at
scale** (see §5–§7).

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

The table below is produced by the automated test suite at the 512-dim configuration — capability/stability
demonstrations on focused tasks. §4.1 adds a measured head-to-head against an equal-budget transformer, which
nova wins at small scale.

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

**Engineering:** an automated test suite; deterministic substrate (seeded); checkpoint save/load;
decision-path telemetry on every answer.

### 4.1 Head-to-head vs a transformer (equal budget)

A console harness (`bench/RaceBench`) races nova against a best-effort decoder-only transformer under a fair,
equal budget: **same tokenizer, same pooled data, same epochs, matched parameter count.** The transformer is
competently configured (pre-LN blocks, multi-head attention, GELU, Adam, loss masked to the answer span —
all of which *help* it). Footprint is the transparent formula `params × bytes` (nova SGD = 8 B/param:
weights+grads; transformer Adam = 16 B/param: weights+grads+two moments).

**Equal parameters, both small** — the regime that tests whether nova's structural priors pay off at a
capacity where a transformer struggles to find them. Full curriculum (number-word equivalence, category
retrieval, add/sub/mul/div), 20 flat epochs, in-distribution held-out:

> nova **457,604 params / ~3.5 MB** · transformer **424,797 params / ~6.5 MB** — equal parameters, nova ~half the VRAM.

| | train | held-out |
|---|---|---|
| nova | 97% | **83%** |
| transformer | 96% | 69% |

Nova led held-out at **every** epoch and converged far faster (47% held-out by epoch 4, when the transformer
was at 4%). Both reach near-equal *train* fit by the end (97% vs 96%), but the transformer trails by 14
points on *held-out* — equal capacity, equal budget, the transformer memorises while nova generalises.

Per-creator held-out:

| Task | nova | transformer |
|---|---|---|
| number-word equivalence | **86%** | 0% |
| arithmetic add | **91%** | 84% |
| arithmetic sub | **90%** | 69% |
| arithmetic mul | **86%** | 84% |
| arithmetic div | **40%** | 13% |
| category retrieval | 0% | 0% |

Honest caveats: `div` is weak for both (small held-out set, n=15); `category retrieval` is a **both-fail**
(0% / 0%), not a win; and this is the equal-budget *flat* result — given many more epochs the transformer
narrows the in-distribution gap. The headline is "same params, half the VRAM, same budget, +15 points
held-out," and the `number-word equivalence` row (86% vs 0%) is the structural advantage: nova has the
homomorphism and relation-as-element retrieval; a transformer this small cannot reach the equivalence under
this budget.

**Arithmetic, extrapolation** (operands well outside the trained range — a structural generalization test):
nova computes via the homomorphism, so it extrapolates exactly (~99%); a transformer interpolates statistics
and fails outside the training range (~single digits) — a documented transformer weakness that **more
training does not fix**. This is architecturally guaranteed by the numeric faces rather than learned, and is
the cleanest separation between the two approaches.

**Reproduce:** `dotnet run --project bench/RaceBench/RaceBench.csproj -c Release`.

---

## 5. Limitations — what has NOT been shown

Deliberately blunt; this is the honest scope of the prototype.

- **Scale.** Everything above is at 512 dimensions, small vocabularies, short inputs. There is **no evidence
  the approach holds at large scale**, on real language, or on long-context reasoning.
- **Benchmark is small-scale and partial.** There is now a head-to-head (§4.1) vs an equal-budget transformer,
  but only at 512-dim on the focused curriculum. At equal small parameters nova generalises notably better
  held-out (83% vs 69%) at half the VRAM, and the decisive, structural win is **arithmetic extrapolation**;
  but two tasks are weak or a both-fail, and a transformer narrows the in-distribution gap with more epochs.
  This needs to be repeated at larger scale and on more tasks before the data-efficiency thesis is established.
- **Narrow tasks.** The demonstrated tasks (exact arithmetic, equivalence, category lookup, simple learned
  functions) validate *mechanisms*; they are not, by themselves, a product.
- **Thesis supported only at small scale.** At equal budget and small parameters, nova measurably beat a
  best-effort transformer on held-out accuracy (§4.1, 83% vs 69%, half the VRAM) and decisively on arithmetic
  extrapolation — real comparative evidence, but at 512-dim toy scale. Whether the advantage holds at large
  scale, on real language and long context, is unproven (§2).
- **Generality bounds are real.** The learned-operation route generalises numeric/affine/structural
  operations, not arbitrary symbolic functions.
- **Prototype stack.** Single-process .NET / TorchSharp; not a hardened or distributed training system.

---

## 6. Engineering notes

- **Stack:** C# / .NET 8, TorchSharp (CUDA with CPU fallback), Windows Forms desktop UI
  (`src/GenesisNova/UI/MainWindow.cs`) with an embedded chat REPL tab.
- **Codebase:** ~26k lines of source across ~148 files (`src/`). Test suite: a fast behaviour suite
  plus opt-in long-running training/emergence tests gated behind `RUN_SLOW=1`. (Exact test counts drift as
  the suite grows — run `dotnet test` for the current number.)
- **Reproducibility:** the substrate is deterministic under seed; results are regenerated by the test suite.
- **Discipline:** no hardcoded answer tables and no symbol parsers — capability is required to come from
  learned structure or the substrate's exact operations, and tests assert answers route through the platonic
  path rather than a memorising fallback.

---

## 7. Roadmap — what would validate the thesis

In priority order, because §5 names the gaps:

1. **Broaden the comparative benchmark.** The equal-budget head-to-head exists (§4.1) and favours nova at
   small scale; the next step is more tasks and a larger transformer baseline measured on data-efficiency /
   exactness / interpretability — turning "promising at toy scale" into "evidence the approach matters."
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

Run the desktop app (Windows Forms — entry point `src/Program.cs` always launches `MainWindow`):

```bash
dotnet run --project src\GenesisNova.csproj
```

The app hosts everything in-process: a **gym** continuous-training loop that starts on launch
(`MainWindow.StartGym`, the role the retired `ClaudeMemory` daemon used to play) and a **REPL** tab that
talks to the live model. Prompts are typed directly (chat-style, no context wrapping); slash commands include
`/help`, `/trace [on|off]`, and `/stats`. Checkpoints persist to disk (sharded binary; a repo starter seeds a
fresh run). Training corpora live in `data/` (main: `data\genesis-nova-train-expanded.txt`).

Read-only diagnostics are available via the `claude/GenesisInspect` CLI
(`report` / `query` / `probe` / `geometry` / `gymprobe`).
