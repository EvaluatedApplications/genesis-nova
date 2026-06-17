# The Platonic Introspection Machine

> From a pattern-predictor that is *managed* to a controller that *manages itself*: a GRU that perceives its
> own platonic space and decision state, holds a **knob per route choice**, and **learns the control policy**
> over its own substrate dynamics — instead of a human twiddling constants.
>
> Status: design / research direction (2026). Successor to `SPACE_AWARE_GRU.md` (perceive→decide→act→verify):
> that paper gave the GRU *eyes*; this one gives it *hands on its own dials* and a reason to learn to use them.

---

## 0. Thesis

Genesis-Nova's GRU is a **thin selector** over a computing substrate (faces, relations, gliders, transforms).
We have made its *decisions* space-aware — route / plan / op heads now read a perception vector
(`ComputeRoutePerception`). But the **dynamics that shape the substrate are still hand-tuned constants**, and
*we* are the meta-controller: this session we measured a collapsed space (`Separation = −0.077`) and **hand-set**
`RepulsionInterval 16→4`, `RepulsionSamples 5→10`, `RepulsionRatio 0.1→0.5` to fix it. That worked — and that is
exactly the problem. The human is in the loop tuning the substrate's physics.

The goal is not better constants. The goal is to **give the GRU the knobs and the introspective signal to set
them itself**, and to **learn**, from the outcome, how to keep its own space healthy — how to categorize and
store knowledge efficiently, how hard to pull related things together and push unrelated apart, when to trust
geometry vs relations, where and how strongly to edit. A model that *predicts patterns* is a reflex; a model
that *knows itself well enough to manipulate itself* is an agent that manages a substrate. That second thing is
the platonic interface we set out to train (see `README.md`, the `nova-north-star` memory).

---

## 1. The problem: a reflex over a hand-tuned physics

Two distinct things are conflated when we say "the GRU learns":

1. **Selection learning (real, bounded).** The heads classify *which* mechanism to use — route, plan/shape, op.
   Gradient-trained on structure-derived labels. This is genuine but small: the GRU is a few logits per head.
2. **Substrate dynamics (hand-set, not learned).** *How* the space evolves is governed by constants a human
   chose. A non-exhaustive inventory of the dials we currently hand-turn:

| Dial (today a `const`) | Where | What it controls |
|---|---|---|
| `GeometryLearningRate` (0.04) | `PlatonicSpaceMemory` | attraction step (pull related together) |
| `RepulsionInterval / Samples / Rate / Ratio` | `ApplyContrastiveRepulsionPass` | push unrelated apart (we just hand-tuned all four) |
| complement-repel threshold (1.35) | `MessagePassUpdate` | dual-space separation |
| `RepelNeighbors` (3) | edit path | how many distractors to repel |
| `GeometricMinConfidence` (0.55), `RelationFirstMinConfidence` (0.5) | inference routes | geometry-vs-relation gating |
| `RebuildSpaceFraction` / `RebuildMinMutations` | `PlatonicLattice` | when to rebuild the spatial index |
| `PlanLossWeight`, `QueryLossWeight`, `RouteLossWeight` | training | head supervision balance |
| edit magnitude prior, mastery bar, difficulty ramp | trainer / gym | how aggressively to write / advance |

Every one of these is a place where **a human's judgement stands in for the model's**. They are also coupled and
context-dependent: the *right* repulsion in a sparse region is wrong in a dense hub; the right geometry-trust for
a well-separated cluster is wrong for a collapsed one. A single global constant is a compromise that is wrong
almost everywhere — which is precisely why the space collapsed silently while relations masked it.

**The reflex problem.** Worse, the GRU *predicts* its outputs (token decode, head argmax) as a stateless
function of the input. It does not look at the consequences of its writes, nor at the health of the region it is
editing. It cannot recover from a bad write because it never reads what it wrote (the `SPACE_AWARE_GRU.md`
finding). Prediction of patterns is the opposite of self-knowledge.

---

## 2. The shift: predict → introspect → manipulate → manage

Frame the whole thing as a POMDP whose **action space includes the controller's own dials**:

| POMDP element | Genesis-Nova realisation |
|---|---|
| **state** `s` | the platonic space (geometry + relations) **and the current knob settings** |
| **observation** `o` | **introspection**: space-health stats + per-route reliability + self-uncertainty (§3) |
| **action** `a` | substrate edit (where / how-strong) **+ knob settings** (forces, thresholds, cadence) (§4) |
| **reward** `r` | did the space get *healthier*: separation ↑, retrievability ↑, mastery held, collapse ↓ |
| **policy** `π` | the GRU heads + new **knob heads**, conditioned on `o` |

The new ingredient versus `SPACE_AWARE_GRU.md` is that **the dials are actions**. The GRU does not merely act
*within* a fixed physics; it *sets the physics*, conditioned on what it perceives about itself. Learning `π` is
learning **to manage its own substrate** — the literal content of "learning the rules of the platonic space."

---

## 3. Knowing itself: the introspection signal

Self-management needs self-observation richer than the current 6 hand-picked scalars
(`[has-neighbour, nearest-conf, degree-norm, mean-top-conf, transform-reliability, bias]`). Introspection has
three tiers:

**(a) Space health (global + local).** The signals we already compute for diagnostics become *inputs*:
- **separation** (related vs unrelated mean distance — `SummarizePushPullGeometry`): is the space collapsing?
- **local density / nearest-distractor distance**: is this region colliding?
- **hub-degree**: is the anchor (or its neighbour) an overloaded hub that will collapse if pulled?
- **drift since last rebuild**: is the index stale?

A controller that *sees* `separation < 0` can *decide* to push harder — the exact judgement we made by hand.

**(b) Per-route reliability (does my tool work here?).** UCB success per route already exists
(`TransformReliabilityRouting` bubbles `BestReliabilityUcb` into route perception). Generalise it: a reliability
estimate per *route* and per *shape*, so the controller perceives "geometry is trustworthy for this cue" vs
"fall to relations," and gates itself (`GeometricMinConfidence` becomes *perceived*, not constant).

**(c) Self-uncertainty (what don't I know?).** Head entropy / margin (is the plan choice confident?), a
forgetting/interference signal (is a previously-mastered skill slipping?), and a "this region is unfamiliar"
flag. These tell the controller *when to write*, *when to abstain*, and *what to rehearse*.

The upgrade path (per `SPACE_AWARE_GRU.md` §C/F): from hand-picked scalars to a **differentiable attention
readout** over the anchor's neighbourhood (and its nearest *distractor*, not just the target), so gradients
shape *what the GRU reads*, keeping it thin (it consumes a summary; the space holds the content).

---

## 4. Manipulating itself: a knob per route choice

The design pattern is uniform and small: **each hand-tuned constant becomes a bounded head output, conditioned
on introspection, centered on the current constant as its prior.** A knob is `clamp(centerᵢ · exp(headᵢ(o)),
loᵢ, hiᵢ)` — the model can only *scale* a safe default within a bounded range, never set an arbitrary value.
This is the anti-divergence discipline: the constant is the safe center; the GRU learns the *deviation* the
local state warrants.

Grouped by the decision they serve:

- **Geometry knobs** (the ones we just twiddled): attraction rate, repulsion rate/ratio/cadence, repel-count.
  Conditioned on `separation`, local density, hub-degree → "collapsing → push; sparse → ease off; hub neighbour
  → damp the pull so items don't collapse onto the hub."
- **Edit knobs**: magnitude *and* **where** (a pointer head attending over candidate concepts — `SPACE_AWARE_GRU.md`
  §D) *and* how-many-to-repel. Conditioned on read-before-write gap → "retrieved-but-wrong → separate the
  winner; empty → create; already-correct → skip."
- **Route-gating knobs**: geometry-vs-relation threshold, platonic-vs-neural temperature. Conditioned on
  perceived retrievability → the known router-erosion failure becomes a *perception* problem the GRU is trained
  on, not a constant we defend.
- **Maintenance knobs**: rebuild cadence from drift; mastery/advance from held stability.

The result: the dials in the §1 table stop being our job. They become the model's policy, set per-context from
introspection.

---

## 5. Learning to manage (not twiddle)

The point is emphatically **not** to expose knobs for a human (or an outer search) to tune. It is for the GRU to
**learn the control policy by outcome**:

- **Reward = space-health delta.** We already compute the within-step retrievability delta
  (`RewardEditHead`); add the **separation delta** (`SummarizePushPullGeometry` before/after a maintenance
  window) and **mastery retention**. The knob heads are trained to *increase* these.
- **Two stable training signals** (mirroring what's already wired):
  - **Lookahead-supervised** (start here — no RL variance): probe a few knob settings on a held-out slice,
    regress the head toward the one that most improved the metric (`SPACE_AWARE_GRU.md` §E generalised from edit
    magnitude to *all* knobs).
  - **REINFORCE** (once perception is informative): the existing `ReinforceEditHead` / `ReinforceRouteHead`
    pattern, extended to knob heads, reward = health delta, **low meta-learning-rate**, bounded step.
- **Bootstrap from the hand-tuned prior.** Today's constants are not thrown away — they are the *center* of each
  knob's range and the initial policy. The model departs from them only where introspection + reward justify it.
  (So the fix we made this session becomes the *prior*, and the GRU learns where to deviate from it.)

The loop: **introspect → set knobs + edit → run a window → measure health → reward → update.** That is a model
managing its substrate, and it compounds without us in the loop.

---

## 6. Why introspection beats prediction

- **A predictor is stateless about consequences.** It maps input→output and never asks "did that help the
  space?" It collides (`acc 0 / rt 100`), over-generates frequent tokens, and cannot undo a bad write. It is a
  reflex.
- **An introspective controller closes the loop.** State (the space + its own knobs) is observable; the edit and
  the dials are actions; "is the space healthier?" is the reward. Self-regulation becomes possible: it can
  notice a collapse and counter it, notice a hub and damp it, notice a stale index and rebuild — *the things we
  did by hand this session.*
- **Self-knowledge is the prerequisite for self-manipulation.** You cannot manage what you cannot see. The GRU
  must *know itself* — its space's health, its routes' reliability, its own uncertainty — *before* it can
  manipulate itself usefully. Introspection is not a feature; it is the precondition for autonomy.
- **This is the north-star, made operational.** "Learning the rules of the platonic space" = learning the
  *control policy over its own dynamics*. Categorizing and storing knowledge efficiently = the learned edit +
  geometry knobs producing clean, separated, retrievable structure. The human's role shifts from *twiddling
  constants* to *defining the reward* (what "healthy" means) and *the guardrails*.

---

## 7. Guardrails & invariants (a self-modifying controller is dangerous)

- **Bounded knobs, prior-centered.** Every knob is `clamp(center · exp(head), lo, hi)`. The model can scale a
  safe default, not invent values. No knob may disable a safety mechanism (e.g. repulsion can't go to 0).
- **Frozen identity is sacrosanct.** No knob, edit, or gradient may perturb the frozen identity dims — the
  numeric homomorphism (poly/log) and char spelling must stay exact (`FaceLayout`, `RestoreFrozenIdentity`).
  Introspection *reads* them; self-manipulation never *writes* them.
- **Meta-stability.** A controller tuning its own learning dynamics can diverge. Mitigations: low meta-LR,
  bounded per-step knob change, start lookahead (not RL), and never let a knob feed back into its *own* update
  rate.
- **Measurement is the reward AND the gate.** The diagnostic tooling is load-bearing: `geometry` (separation),
  `GenesisInspect probe` (router erosion), the gate tests (`PlatonicGeometryDynamicsTests`). A learned knob ships
  only if it **matches or beats the hand-tuned constant** on the metric, with no regression on the fast suite —
  otherwise the constant stands.
- **Test-first, demonstrate-can-emerge.** Each knob is introduced behind its own emergence test: prove the
  *learned* policy reaches the hand-tuned baseline before trusting it to exceed it.

---

## 8. Roadmap (concrete, measurement-gated increments)

0. **Prerequisite — un-collapse + retrain.** Eyes on a collapsed space see noise; perception only pays off once
   the geometry separates. (Forces fixed + space reset this session; confirm `geometry` separation goes positive
   on a retrain *first*.)
1. **Introspection v2.** Extend the perception vector with space-health stats (separation, collapse indicator,
   hub-degree, drift) — the richer eyes. No behaviour change yet; just expose the signal.
2. **One learned knob, end-to-end.** Make **repulsion strength** a knob conditioned on the collapse indicator,
   trained (lookahead) on the separation delta, gated by `PlatonicGeometryDynamicsTests` (learned ≥ hand-tuned).
   This proves the whole loop on the dial we *just hand-tuned* — the cleanest possible demonstration.
3. **Route-gating knobs.** `GeometricMinConfidence` / relation-vs-geometry from perceived retrievability —
   attack router erosion as a learned-gating problem.
4. **Edit placement.** Pointer head (where to write) + repel-count, from read-before-write introspection — the
   "fine-edit its skills" capability.
5. **The perception curriculum.** Atomic *read-the-space* lessons ("is X related to Y?", "what's nearest X?",
   "does X retrieve Y yet?" → "make X retrieve Y") so introspection is *meaningful* before the write policy
   exploits it (`SPACE_AWARE_GRU.md` §H). This is the most direct path to "learning the rules of the space."
6. **Generalise to the remaining dials** (geometry rates, cadence, loss weights) only as each earns its keep.

---

## 9. The end state

A GRU that:
- **reads its own space and decision state** (introspection v2 + attention readout),
- **holds a bounded knob for each route/edit choice** (forces, thresholds, placement, cadence),
- **learns the control policy by outcome** (separation / retrievability / retention deltas, lookahead then
  REINFORCE, prior-centered and bounded),
- and therefore **categorizes and stores knowledge efficiently because it has learned to** — not because we
  hand-set the physics.

It stops predicting patterns and starts **managing a substrate it understands**. Our job becomes defining what a
healthy space *is* (the reward) and the invariants it must never break (the guardrails) — and then getting out
of the loop. That is the platonic introspection machine.

---

*Related: `SPACE_AWARE_GRU.md` (perception/edit foundation), `PLATONIC_SPACE.md` (substrate), `PLATONIC_SHAPES.md`
(composer), `claude/SELF_IMPROVE.md` (the compounding loop), and the memories `nova-space-aware-gru`,
`nova-pushpull-tuning` (the hand-tuned fix this paper proposes to make learned), `nova-reliability-routing`,
`nova-north-star`.*
