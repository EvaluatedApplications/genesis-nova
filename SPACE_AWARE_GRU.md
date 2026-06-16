# Space-Aware GRU — training the controller to READ its platonic space before it acts

> Status: design / research ideas (2026-06-16). Novel directions for making the GRU **perceive** the current
> contents of the platonic space and **condition its actions on what it sees**, rather than choosing actions
> blind from the input tokens alone.

## The problem in one sentence

Today the GRU decides **what to do to the space** (route, edit-magnitude, plan/glider) as a feed-forward
function of the **input tokens only** — it never *looks at what is already in the space*. It writes blind. The
user's framing of the fix:

> it has to learn: **(training pairs) → (look at the platonic space) → (modify the space to best get the right
> answer)**. It needs to *read the space* rather than randomly choosing actions.

That is a shift from a **stateless reflex** to a **state-aware policy**: the platonic space is the *observable
state*, a modification is the *action*, and "does the space now produce the target?" is the *reward*.

## What exists today (the blind spots)

Grounding in the current engine (see `PLATONIC_SPACE.md`, `Train/GenesisTrainer.cs`,
`Cognition/PlatonicSpaceMemory.cs`, `Model/GenesisNeuralModel.cs`):

- **`PredictEditMagnitude(inputTokens)`** — the edit head predicts *how strongly to write* from the input
  tokens only. It does **not** see the anchor's current neighbours, edge strengths, or whether the target is
  already retrievable. Blind.
- **Route head / `ResolveRouteLabel`** — the platonic route is *supervised* (offline) only once the space can
  already retrieve the target, but at **runtime** the GRU's route choice doesn't read a live "can the space
  answer this?" signal; it infers from tokens and falls back when a tool yields nothing.
- **`ObservePlatonicSpace`** — forms the cue→answer edge with a *deterministic mirror* coupling; the *direction*
  of the edit is hand-coded (toward the nearest input concept), not chosen by a controller that perceived the
  region.
- **`GetNearestConcepts` / `QueryConceptChain`** — the space CAN be read, and retrieval uses it — but that read
  is **not fed back into the GRU's decision** about what/where/how to edit. Perception and action are decoupled.
- The recent **within-step edit reward** (`RewardEditHead`: post−pre retrievability delta) is a step toward
  closing the loop on the *reward* side, but the *action* is still chosen without perceiving the pre-state.

Net: the controller is a reflex. We want a loop.

## The paradigm: perceive → decide → act → verify

Frame one training step as a tiny POMDP over the space:

1. **Observe** `o = Perceive(space, cue)` — a fixed-width readout of the *relevant region* of the space for this
   cue (what's here, how strong, is the target near, what are the distractors).
2. **Decide** `a = π_GRU(tokens, o)` — the heads (route / edit / plan / pointer) now condition on `o`, not just
   tokens.
3. **Act** — apply `a` to the space (write/relate/compose/route).
4. **Verify** `r = Retrieve(space', cue) ≟ target` — reward = the retrievability *gain* the action produced
   (the within-step delta we already compute).

The GRU learns `π` to maximise `r`. The key new ingredient is `o` — **perception of the space** as an input to
every decision.

---

## Novel mechanisms

### A. Space-perception vector (the core: give the GRU eyes)
Before any head fires, compute a small, fixed-width **perception vector** for the cue's anchor concept(s) and
concatenate it to the GRU's hidden input (`hInput`). Candidate features (all cheap reads of existing structure):
- the anchor's **degree** and mean edge strength (is it a hub? overloaded?),
- **is the target already the nearest neighbour?** (rank + distance of the best current candidate),
- **local density / nearest-distractor distance** (is this region colliding?),
- the **retrieval confidence** the current space would produce for this cue,
- a pooled embedding of the anchor's **k nearest neighbours** (what's actually around it).

Now the edit head can learn "this anchor is already a dense hub at rank-1 for the wrong concept → push the
distractor away" vs "empty region → create a fresh strong edge." That decision is *impossible* today.

### B. Read-before-write loop (the literal user request)
Make the step **two-phase**:
1. **Read**: run the retrieval the model *would* answer with; observe `(retrieved, confidence, gap-to-target)`.
2. **Write**: condition the edit on that gap — **skip** if already correct (we have skip-correct, but make it a
   *perceived* decision, not an offline label); **separate** if retrieved-but-wrong; **create** if empty.
This turns editing from "always nudge toward the mirror" into "act on the observed error," which is exactly how
a person edits a note: look at what's there, then change what's wrong.

### C. Differentiable space readout (attention pool over neighbours)
A learned **space-encoder**: attention over the anchor's neighbourhood embeddings → a differentiable summary
fed to the GRU. Unlike the hand-picked features in (A), this lets gradients shape *what* the GRU reads. Keeps
the GRU thin (it consumes a summary; the space holds the content) per the project's "thin selector" principle.

### D. Pointer-attention into the space (choose WHERE to act)
Instead of always editing the cue's mirror-nearest concept, give the GRU a **pointer head** that attends over
candidate concepts and *selects which to relate / separate / strengthen*, conditioned on perception. The space
becomes addressable content-wise (cf. Neural Turing Machine / DNC content addressing) and the controller learns
*where* to write, not just *how hard*.

### E. Gap-minimising edit policy (lookahead-supervised)
Extend the within-step reward into a **supervised target** for the edit head: probe a few candidate
(direction, magnitude) edits, measure which most reduces the retrieval gap, and train the head toward it
(1-step lookahead / learned value). The head stops guessing magnitude and learns the *edit that demonstrably
helps*, because it now perceives the pre-state to predict the post-state.

### F. Contrastive perception (see the distractors, not just the target)
Feed the GRU the **nearest wrong neighbours** of the anchor, so it can choose to *repel* the specific confuser
that's currently winning. This directly attacks the collision failure mode (`acc 0 / rt 100`) and pairs with the
contrastive edit-outcome we already compute. Perception of "who am I being confused with" is actionable.

### G. Memory-augmented framing (NTM / DNC analogy)
The cleanest mental model: the platonic space is an **external differentiable memory**; the GRU is a controller
with learned **read** (content-based addressing = `GetNearestConcepts`) and **write** (the edit/relate ops)
heads, where the write is conditioned on the read. The decades of memory-augmented-network results say this is
the right shape — and it's exactly "read the space, then modify it."

### H. Explicit perception curriculum (teach reading before writing)
Atomic lessons that *require* reading the space to answer, trained alongside the index (use the
`AssociativeIndexLearningTests` harness):
- **probe**: "is X already related to Y?" → yes/no (the model must read an edge),
- **neighbour**: "what is nearest X?" → name (read the geometry),
- **gap**: "does X retrieve Y yet?" → confidence,
- **then act**: "make X retrieve Y" → write.
Mastering the *read* lessons gives the perception vector meaning the *write* policy can exploit. This is the
"atomically teach it how to construct these" idea applied to the read/write skills themselves.

### I. Perceived confidence → routing & abstention
The route head should pick platonic-vs-neural from a **live perceived retrievability** signal (part of `o`),
not an offline correctness label. "The space can answer this (high perceived confidence) → route platonic;
it can't → either build it or fall back." Turns the known router-erosion failure into a *perception* problem the
GRU can be trained on directly.

---

## Training the loop

| POMDP element | Genesis-nova realisation |
|---|---|
| **state** `s` | the platonic space (geometry + relation graph) |
| **observation** `o` | perception vector (A) / attention readout (C) for the cue region |
| **action** `a` | route choice + edit (direction, magnitude) + pointer target + plan/glider |
| **reward** `r` | within-step retrievability delta (post−pre), contrastive + bidirectional (already built) |
| **policy** `π` | the GRU heads, now conditioned on `o` |

Two compatible training signals:
- **Supervised (lookahead)**: derive the *best* action by probing (E) and regress the head toward it — stable,
  no high-variance RL.
- **REINFORCE (policy-gradient)**: the existing `ReinforceEditHead` path, but with `o` in the state and the
  within-step delta as reward — now the gradient can credit *reading* the space.

Start supervised (lookahead) to avoid RL variance; add policy-gradient once perception is informative.

## First experiment (test-first, cheap)

In `Tests/AssociativeIndexLearningTests.cs` (the controlled index harness that already shows clean broad
training → 100%):
1. Add a **perception feature** = "is the target currently the anchor's nearest neighbour, and at what rank?"
   exposed to the edit head.
2. Lesson: deliberately **pre-poison** an anchor with a strong wrong edge, then train. **Hypothesis:** a
   space-aware edit head *separates the distractor and recovers*, while the blind head cannot (it only attracts
   toward the mirror). Measure recovery rate blind-vs-aware.
3. Success = aware ≫ blind on poisoned recovery, with **no regression** on the clean index or the 107-test suite
   (arithmetic homomorphism + retention are the landmines — never let perception touch frozen identity dims).

If that holds, graduate the perception vector into `GenesisNeuralModel` (concat to `hInput`) behind a config
flag, and A/B it on the daemon's real index.

## Guardrails

- **Keep the GRU thin** — it consumes a *summary* of the space; the space stays the substrate (project axiom).
- **Never let perception or its gradients perturb frozen identity dims** — the numeric homomorphism (poly/log)
  and char spelling must stay exact (see `[[nova-relational-geometry]]`, `FaceLayout`).
- **Test-first, demonstrate-can-emerge** — write the emergence/poison-recovery test first; assert a majority bar
  AND that it routes via the platonic path; guard the 107-test suite (arithmetic-router erosion is the known
  regression, see `[[nova-generalized-training-routing]]`).
- **Empirically grounded** — use the harness to prove each perception feature *earns its keep* before it goes
  into the core (see `[[nova-index-learning-empirics]]`, `[[first-principles-not-test-fitting]]`).

## Why this is the right direction

Everything we've learned this session points here: the engine *constructs* associations perfectly when taught
cleanly, but it **edits blind** — so it collides (`acc 0 / rt 100`), over-generates the frequent tokens, and
can't *recover* from a bad write because it never looks at what it wrote. Giving the controller **perception of
its own space** is the difference between a reflex and an agent that reads, reasons about what's there, and acts
to fix it — which is the whole mission of training a platonic *interface*, not an LLM (see `PLATONIC_SPACE.md`,
`PROJECT_GLIDER.md`, the `nova-north-star` memory).
