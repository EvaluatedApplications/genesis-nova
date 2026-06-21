# Backpropagating the Platonic Space

> The platonic space is the only part of Genesis-Nova that is **not** trained by gradient. The GRU heads
> descend a CE loss; the space is moved by **hand-coded forces** (attraction + contrastive repulsion) with
> hand-set constants — the dials we keep twiddling. This paper asks: how do we let a *gradient* shape the space
> instead, and what does that buy us.
>
> Status: design / research (2026). Companion to `SPACE_AWARE_GRU.md` (perception) and
> `PLATONIC_INTROSPECTION.md` (learned knobs). This goes further than both: knobs *tune* the hand-forces;
> backprop *removes* them.

---

## 0. Thesis

The hand-force approach has a measured ceiling. After fixing the frozen-number dilution and adding an attraction
floor, the live space separates at only **+0.04** (related 0.467 vs unrelated 0.508) while a clean synthetic
gate hits **+0.47**. The gap is structural: **greedy, per-pair, local forces cannot jointly optimize a tangled,
high-degree graph** (a hub pulled toward six items lands at their centroid, ~0.47 from each, not tight). A
*global* objective optimized by gradient does not have this problem. So: make the space's free dimensions
differentiable, define the geometry we want as a **loss**, and let backprop find the configuration the forces
can't.

---

## 1. Why the space isn't backprop'd today — the three walls

1. **Bridged by name, not by tensor.** Concept→token-bias is a hash/lookup (`argmax`); no gradient flows from
   the decoder's loss into a concept's face. The space sits *outside* the autograd graph.
2. **Retrieval is `argmin`.** Nearest-concept (VP-Tree) and relation hops are discrete — non-differentiable.
3. **Frozen identity.** Poly/log (number value) and char (spelling) must stay exact; any gradient must be
   *masked* off them.

None of these is fatal; each has a known dissolve (below).

---

## 2. Two different gradients we actually want

- **A geometry gradient** — "pull related close, push unrelated far." Today: `MessagePassUpdate` +
  `ApplyContrastiveRepulsionPass`, hand-tuned (`RepulsionRatio`, `MinAttractAffinity`, the dilution fix…).
- **A function gradient** — "shape the space so it produces the *right answer*." Today: **no path exists** for
  the end-task loss to reach the space at all.

They need different mechanisms (§4 and §5). But before either, there is a structural fix that makes the current
forces sane *and* is the gradient-free precursor to the contrastive loss.

---

## 3. The pull/push inconsistency — and the unification that fixes it

**The catch (correct and important):** the **pull relies on edges** — attraction runs over the relation graph's
actual related pairs (`UpdateConceptGeometry` per `ObserveContradiction`). The **push does NOT** — repulsion
(`ApplyContrastiveRepulsionPass`) samples **random non-edged** concepts. One half is exhaustive and structural;
the other is stochastic and edgeless. That asymmetry is exactly why:

- **It under-covers at scale.** A concept has a handful of edges (all pulled) but thousands of non-edges (only
  ~10 sampled). The push touches a vanishing fraction → weak, noisy separation; and it *wasted* most samples on
  frozen numbers until we patched it (the dilution bug was a *symptom* of the edgeless random push).
- **It pushes the wrong things.** Random negatives are mostly already-far concepts. The pair that actually needs
  separating is the **near non-edged confuser** — the thing colliding with this concept *right now* — which a
  uniform random sample almost never hits.

**The unification:** anchor BOTH halves on the same object. For each **edge `(a,b)`** (a positive), in the same
step:
- **pull** `a`→`b` (as now), and
- **push** `a` away from its **nearest NON-edged neighbours** — the *hard negatives*, the concepts currently
  colliding with `a` (found via the lattice we already have), not random draws.

Now pull and push are consistent: both are anchored on a concept's *real* relationships and its *real*
neighbourhood. This is **hard-negative mining**, and it is far more effective per update than uniform random
repulsion — and it is **exactly the InfoNCE structure** (positive = the edge; negatives = the nearby non-edges).
So fixing the inconsistency *is* building the gradient-free skeleton of the backprop loss. Do this first; it
should already lift the +0.04 ceiling, and it makes the jump to §4 a one-line swap (replace the manual nudge
with `loss.backward()`).

---

## 4. Mechanism 1 — the geometry gradient *is* a contrastive loss (and it deletes the constants)

Make each concept's **free/semantic dims** a masked `torch.Parameter` (identity dims detached → numbers/spelling
cannot move and receive *zero* gradient — so the dilution class of bug is structurally impossible). Use the
relation graph as supervision: each edge is a **positive**; its nearest non-edged neighbours are **hard
negatives** (§3). Minimize **InfoNCE / triplet** over semantic-face distances and `backward()`.

The punchline: **the gradient of InfoNCE *is* "pull the positive, push the hard negatives," self-scaled by how
wrong each pair is.** Everything we hand-tuned — repulsion strength, the dilution patch, the attraction floor —
becomes *emergent from the gradient*. There is no `RepulsionRatio` to set. And because it optimizes the **whole
configuration jointly**, it resolves the tangled-hub case the greedy forces can't (the hub finds a position that
trades off all its edges *and* all its collisions at once). Relations stay primary: the **graph is the
supervision** (which things relate); the **gradient is the geometry** (where they sit) — an EM-style split.

---

## 5. Mechanism 2 — the function gradient via a *soft, differentiable retrieval bridge*

Replace the hard concept→token-bias lookup with a **differentiable attention read**: `softmax(hidden · faces)`
→ a weighted bias over tokens. Now the **decoder's answer-loss backprops *through* the read *into* the faces** —
the space is trained by whether it *helped produce the right answer*, jointly with the GRU, under one loss. The
space learns to be retrievable *because it is useful*. This is the real unification: the two halves stop being
name-bridged and become one autograd graph.

---

## 6. Mechanism 3 — a gradient through the hard `argmin` (inference unchanged)

Don't ship soft-attention (O(N), fuzzy). Use a **straight-through estimator**: forward = hard KNN/argmin (exact,
fast, what ships); backward = the softmax surrogate's gradient. Inference behaviour is byte-identical; training
still gets a signal into the faces.

---

## 7. Mechanism 4 (ambitious) — backprop through *where the space settles*

The space is a **fixed point** of its dynamics. Borrow **Deep Equilibrium Models**: solve for the equilibrium
positions, then use **implicit differentiation** to backprop a downstream loss through that fixed point *without
unrolling* the iterations — constant memory, regardless of how many message-passing steps it took. This makes
the entire settling process trainable, not just one step of it.

---

## 8. Guardrails

- **Frozen identity by construction.** Only semantic free dims are Parameters; poly/log/char are detached → the
  homomorphism stays exact and gets zero gradient. (This is *stronger* than the current restore-after rule.)
- **Keep the GRU thin.** The soft read is a *training-time* differentiable view; the hard read is what ships.
- **Measurement is the gate.** `SummarizePushPullGeometry` (separation), `GenesisInspect probe` (router
  erosion), `PlatonicGeometryDynamicsTests`. A gradient-shaped space ships only if it **≥ the hand-forced**
  baseline on separation, with no fast-suite regression.
- **Stability.** Low LR on the face Parameters; bounded; the relation graph (discrete) anchors the supervision
  so the geometry can't run away.

---

## 9. First experiment + roadmap

**Experiment (cleanest possible proof):** take a few hundred concept faces as masked Parameters, build positives
(edges) + hard negatives (nearest non-edged) from the live graph, minimize InfoNCE for a few hundred steps, and
read `SummarizePushPullGeometry`. **Question: does pure backprop reach/beat the hand-forced separation — with
zero force constants — on real tangled structure?** If yes, the geometry can be *descended* instead of *tuned*.

**Roadmap:**
0. (now) **Unify pull/push** into per-edge positive + hard-negative (§3) — gradient-free; lifts the ceiling and
   builds the skeleton.
1. **InfoNCE on faces** behind a flag (§4); A/B vs the hand-forces on the separation gate.
2. **Soft-retrieval bridge** (§5) — the function gradient; the space trained by the end-task.
3. **STE** (§6) so inference's hard retrieval keeps a training signal.
4. **Equilibrium diff** (§7) — the full settling process trainable.

The throughline: we have spent three rounds hand-tuning forces and hit a +0.04 ceiling on real structure. The
gradient of a contrastive loss *is* the force, self-tuned and global — and a downstream loss can shape the space
*functionally*. That is the substrate-side realization of "learn the rules of the platonic space."

---

*Related: `PLATONIC_INTROSPECTION.md` (learned knobs — superseded for geometry by this), `SPACE_AWARE_GRU.md`
(perception/edit), `PLATONIC_SPACE.md` (substrate). Memories: `nova-pushpull-tuning` (the hand-force ceiling
this proposes to descend past), `nova-relational-geometry`, `nova-north-star`.*
