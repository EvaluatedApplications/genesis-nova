# GenesisNova — System Overview & the Road to an LLM Competitor

*Status as of 2026-06-30. Empirical, not aspirational — every claim below is backed by a measurement from this codebase.*

> **Update (Phase 1 started):** the "statistics on the GPU" rebalance is no longer purely proposed. The cloud
> recompute — the substrate's hottest scalar loop — now has a batched GPU path (`Cognition/Platonic/GpuCloudBatcher.cs`,
> driven by `DialecticalSpace.BatchedCloudGpu`). It defers + dedups dirty clouds and recomputes them in one
> `index_select`/`index_add_` op on CUDA (Cloud = A·T), validated cosine-identical to the scalar definition. It is
> **gated `default-off`** (`GenesisNovaConfig.BatchedCloudGpu = false`, NOT yet in `WithProductionMechanisms`) and not
> yet wired into the gym loop, so the live default path below is still the scalar CPU loop. Sections 5–7 describe that
> default; the GPU rows in §6 are now partly built rather than hypothetical.

---

## 1. What we are building

**One conscious field**: a structured *platonic substrate* that a *thin neural layer* reasons WITH — not an LLM that stores knowledge in weights.

The bet: meaning has **structure** (composition, relation, homomorphism), and structure should live in an explicit, inspectable, editable substrate — not be smeared across a billion opaque parameters. The neural net's job is to *recognise* structure from surface form and *direct* the substrate; the substrate's job is to *store, compose, and retrieve* meaning.

That bet is **half-right**, and this document is about which half.

---

## 2. The architecture today

In production (`GenesisNovaConfig.WithProductionMechanisms()`, used by the desktop app and RaceBench) the engine runs
with `ConsciousField = true`, and cognition is driven by the **substrate's own field relaxation** — *not* by a GRU
classifier deciding routes. The shipped path is:

```
   SURFACE (text/tokens)
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │  DialecticalSpace (the platonic substrate)    │   ~90% of compute
   │  • Elements (atoms: char / word / number)     │
   │  • Relations (the graph / force-field)        │   GenerateFromField:
   │  • Clouds (distributional geometry)            │   a substrate route LADDER
   │  • Homomorphism codec (exact arithmetic)       │   that abstains / falls
   │  • Relaxation = reasoning (settles surprise)   │   through per step
   └──────────────────────────────────────────────┘
        ▲
        │  thin recogniser (op-cue, role, abstain signals) — NOT a route classifier
   ┌─────────────┐
   │  NN (GRU)   │   ~10% of compute
   └─────────────┘
   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄ LEGACY / aspirational ┄┄┄┄┄┄┄┄┄┄┄┄┄┄
   the GRU route/plan/op CLASSIFIER (`GenerateSingle`) — the old "NN directs"
   path, live only when `ConsciousField = false`; bypassed in production
```

**The shipped cognition path:** with `ConsciousField` on, `GenesisInferenceEngine.GenerateFromField` runs a substrate
route ladder that abstains / falls through per step (induction → predicate → arithmetic → number-word → field-tick →
learned-function → meaning-ticks/analogy/compose → talk → learn → recall → **relaxation (`TryFieldRelax`)** → abstain),
each answer tagged with a `DecisionPath`. Reasoning is the field **settling its own surprise** (the founding claim of
`PLATONIC_MIND.md` §3), with honest abstention when nothing settles — and, in the ambiguous branch, a trained
**navigator walk** (`PLATONIC_NAVIGATOR.md`) before the one-shot relax. The GRU is demoted to a **thin recogniser**
(op-cue, role, abstain signals), not the director.

**The legacy path (bypassed in production):** the GRU route/plan/op **classifier** — the "NN recognises routes/ops and
directs" arrangement — is the `ConsciousField = false` default (`GenerateSingle`). It is the path this document's
critique (§4, the overfitting) is about, and the aspirational recogniser role §6 describes; it is **not** the live
production cognition path.

**What lives where:**
- **Substrate (CPU):** identity, relations, geometric clouds, homomorphic arithmetic, Merge-style composition, recall by relaxation, eviction/forgetting — and, in production, the **route ladder + relaxation that drive cognition**.
- **NN (GPU):** role recognition, op-cue inference, abstain signals — a *general pattern recogniser* that supervises the substrate. (Using it as a route/plan/op *classifier* is the legacy arrangement, off in production.)

---

## 3. What the substrate does WELL (this scales — keep it)

These are **structural / compositional** capabilities. They work because the answer genuinely *has* structure the substrate can represent exactly:

| Capability | Evidence |
|---|---|
| **Homomorphic arithmetic** | Exact decode incl. unseen `84+57=141`; numbers freeze a sub-face and compose in base-10. |
| **Compositional number-words** | Learned word↔value atoms, scales solved compositionally; codec-off 0%→100%. |
| **Merge / relation-as-element** | Binary recursive composition is native to the substrate (Kind.Relation + Components). |
| **Recall by relaxation** | Field settles to the right concept; disambiguation + abstention proven. |
| **Recognition / generalisation** | Novel `helix` positioned from known morphemes `hel-`; nearest = `[hello, greet, help]`. |
| **Eviction by relevance** | Forgetting curve keyed on *contributed-to-answers*, not mere observation; bounded ~4.5k nodes / 33MB, stable overnight. |

**Why it scales:** structural answers are *exact and local*. Composition cost is bounded by the depth of the expression, not the size of the corpus. This is the part of the thesis that is paying off.

---

## 4. What the substrate does POORLY (this is the overfitting — move it)

These are **distributional / statistical** capabilities. They *don't structure cleanly*, and every attempt to force them into the graph hits a ceiling:

| Attempt | Result |
|---|---|
| **Graph-clustering function-word metric** (`NeighbourCoherence`) | **Caps at ~36%.** Possessives (my/your, coherence 0.16–0.27) overlap content (0.22–0.54). No threshold cuts cleanly. |
| **PMI distributional metric** (theory #2) | **FAILED live: −7% separation** (flags 13/14 glue AND 12/12 content). Worse than the graph metric. Inverted on small synthetic vocab. Gated OFF, unwired. |
| **Foundation separation ceiling** | Stuck ~21–36% regardless of metric. The signal is real but *noisy and continuous* — exactly what a graph of discrete relations represents badly. |

**Why it fails:** distributional semantics is a *smooth, high-dimensional, statistical* object. "How function-like is this word" is a soft gradient over co-occurrence statistics — there is no clean discrete relation to store. Forcing it into the graph means hand-tuning thresholds against a specific data regime = **overfitting**. The PMI negative result is the proof: a metric that's *correct in theory* and *correct on clean synthetic data* still fails live, because the real signal lives in the *statistics*, not the *structure*.

This is the user's diagnosis, confirmed: **we are overfitting in the platonic space.**

---

## 5. The compute imbalance (the smoking gun)

Profiling the live training loop:

```
  ~90%  CPU substrate (RecomputeCloud, relation updates, relaxation, eviction)
  ~10%  GPU / NN     (batch=1, sequential with the substrate — no overlap)
```

The GPU is **idle**. The NN runs *batch-of-one, in lockstep* with the substrate — the dual-compute is unrealised. We are doing the **hard statistical work (clouds, kNN, relaxation) on the CPU, one observation at a time**, while a tensor engine built exactly for that sits waiting.

This is backwards. The substrate's *structural* work (homomorphism, Merge, exact recall) is cheap and belongs on CPU. The *distributional* work (clouds, similarity, soft routing) is expensive, batchable, and belongs on GPU.

---

## 6. The strategy: a clean split of labour

> **Structure on the substrate. Statistics on the GPU.**

Move the **hard, doesn't-structure-well, distributional** work off the CPU graph and onto the GPU/NN as **batched tensor operations**, keeping the substrate for what it's exact at.

| Today (CPU substrate, overfit) | Proposed (GPU/NN, batched) |
|---|---|
| `RecomputeCloud` per observation | **Clouds as sparse matvec** — periodic batched recompute on GPU — **BUILT** (`GpuCloudBatcher`, gated `BatchedCloudGpu`, default-off, gym-wiring pending) |
| kNN nearest-concept scan | **kNN as matmul** over the embedding matrix |
| Function-likeness via graph thresholds | **Distributional semantics as a learned GPU embedding / factorization** — the NN *learns* the soft function/content gradient instead of us thresholding a graph |
| Field relaxation, one step at a time | **Relaxation as attention** — batched, parallel settling |
| Op-cue / role recognition (already NN) | Keep, but **batch it** so the GPU is saturated |

**Keep on the substrate (unchanged):** homomorphic codec, Merge/composition, exact recall of stored relations, eviction. These are exact and structural — they don't want a GPU.

**The principle (already in memory as architectural law):** the NN is the *general pattern recogniser* — it handles the fuzzy, hard-to-hardcode statistical job and *generalises*. The substrate is *structural* — it stores, composes, retrieves. We've been making the substrate do the recogniser's job (graph thresholds for a statistical gradient). Stop. Give the statistics back to the NN, on the GPU, in batches.

---

## 7. The road to an LLM competitor

An LLM is, crudely, **one giant distributional engine** — everything (syntax, semantics, facts, reasoning) compressed into smooth statistics. Its strength is *fluency and coverage*; its weakness is *no exact structure* (it can't truly compose, count, or hold a revisable belief — it approximates all of it statistically).

Our differentiated bet: **a hybrid that is exact where structure exists and fluent where it doesn't.**

**Phase 1 — Rebalance (now, in progress).** Fix the 90/10 split. Move clouds/kNN/relaxation/distributional-semantics to batched GPU. Saturate the tensor engine. This alone unblocks scale — the substrate stops being the bottleneck. *Status:* the cloud-recompute piece is built (`GpuCloudBatcher`, defer+dedup+`index_add_` on CUDA, validated cosine-identical), but gated `default-off` and not yet wired into the gym/production loop; kNN and relaxation are still scalar CPU.

**Phase 2 — Let the NN own the statistics.** Stop hand-tuning graph metrics for distributional properties (function words, soft routing, ambiguity). The NN learns them from data on the GPU. The substrate keeps only exact structure. (The PMI failure is the mandate for this.)

**Phase 3 — Scale via diversity, not size.** Distributional semantics *needs* large, diverse data (the PMI experiment proved small synthetic data is the wrong regime). The genesis lineage already points the way: **evolutionary / parallel training — many model variants on varied data slices, then merge.** This is how you get corpus-scale distributional coverage without the averaging collapse that killed raw-corpus training.

**Phase 4 — Fluent generation over the hybrid.** The talk-by-chunk / NN-directed-generative-tick work (already prototyped) becomes the output layer: the NN sequences chunks retrieved from the substrate, producing fluent text that is *grounded in exact structure* where it exists. That's the competitor: GPT-class fluency with a substrate that can actually count, compose, and revise a belief.

---

## 8. The honest risks

- **Phase 1 is real engineering**, not a flag flip. The substrate's cloud/kNN/relaxation paths are written for per-observation CPU; batching them on GPU is a rewrite of the hot loops.
- **The merge step (Phase 3) is unsolved** at our scale — model merging is finicky.
- **We have repeatedly overfit to the synthetic data regime.** Every metric we tune on clean synthetic data risks failing live (PMI did). Phase 2 must be validated on *diverse* data, live, before committing.
- **The substrate must stay the source of exact structure.** If we let statistics leak back into it (more thresholds, more hardcoded lists), we recreate the overfitting we're escaping.

---

## 9. One-line summary

> We built an exact structural mind and then asked it to do fuzzy statistics by hand — that's the overfitting, and the 90/10 CPU/GPU split is its fingerprint. **Move the statistics to the GPU where they're batched and learned; keep the structure on the substrate where it's exact.** That rebalance, plus scale-via-diversity, is the road to a competitor that is fluent *and* can actually compose.
