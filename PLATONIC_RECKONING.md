# The Reckoning — real platonic intelligence vs. overfitting to tests

> Written 2026-06 after a long stretch in the wrong direction. The test for everything below: **does it generalize
> from structure, or does it only work because we built a detector / template / classifier shaped to the tests?**
> This document supersedes the over-claimed parts of `PLATONIC_MIND.md` and `PLATONIC_CONSCIOUSNESS.md`: keep their
> *mechanisms*, hold their *narrative* lightly. It is the practical anchor for what to keep and what to shed.
>
> *Framing.* A platonic space is a **space of ideas**; we may make its rules freely **so long as they obey the
> axioms** (G1–G6, defined in `PLATONIC_THEORY.md` — observer, non-contradiction, conservation, irreversibility:
> logical/mathematical, not physical). Words like *field*, *relaxation*, *attractor*, *free-energy*, *alive*,
> *consciousness* are **generative metaphors that motivate the design** — not claims that the system models physics
> or is conscious. The test is always the engineering one: *does it generalize from structure?* — never the
> metaphor's literal truth.

---

## 0. Status — what has landed since (2026-06-24)

The **seam fix + the first piece of the keep-core control path are built and proven**, behind one flag
(`GenesisNovaConfig.KeepCoreControl`, default-off → byte-identical; the desktop app turns it on):
- **Anchor seam fix** — trainer route-label + perception and inference retrieval now anchor on the *same*
  discriminative cue (shared `Cognition/PlatonicConceptAnchors`). The reckoning's one real seam bug, closed.
- **Relaxation as the retrieval route** — `DialecticalSpace.Reason` wired into the ladder as `platonic-reason`.
- **Abstain over hallucinate** — a non-arithmetic query that nothing settles returns `platonic-abstain`, not a
  neural-decoder guess (with a hub-abstention guard for unknown referents).
- *Proven:* the exact bug scenario (`a synonym for big → large`) → 100% retrieval, 8/8 via relaxation; unknown →
  silent abstain; arithmetic exact. See `[[nova-keepcore-landed]]`.

Also landed (efficiency, not in the original reckoning but needed for scale): the **lattice** (VP-tree) was wired
back into `DialecticalSpace`, which had regressed to O(N) scans — O(log N) candidate-gathering with live-face
rescoring, proven 100% equivalent to the scan and ~9× faster at 440 nodes. See `[[nova-dialectical-lattice]]`.

**Still NOT done — the deeper subtraction (re-verified 2026-06):** removing the PLAN head + 9 shapes, the structural
label resolvers, and rewriting training to self-supervised prediction. Confirmed still pending in code — the throw
items below all still exist: `PlatonicShapeRegistry` + the PLAN head, `ResolvePlanLabel` (`Train/GenesisLabelResolver.cs`),
and `Cognition/Platonic/FieldCognition.cs` / `FieldOperator.cs`. That is the next step (§6–§8 below), to be done with
the user in the loop — not blind. Everything below is the standing plan; §8's "fix in place" is the part now landed
(seam fix verified: `PlatonicConceptAnchors.SelectDiscriminative` is what the trainer anchors on under
`KeepCoreControl` — `Train/GenesisTrainer.cs`, `Cognition/PlatonicConceptAnchors.cs`).

---

## 1. The honest diagnosis

The substrate has genuine, generalizing intelligence. The **orchestration layer grew into an overfit task-classifier**:
every gym skill we ever added spawned a new plan-kind + a structural label-detector + a hand-built glider shape. That
growth *feels* like progress and is the opposite of it — it makes the system better at a fixed menu of ~13 synthetic
tasks (and only ~49% even at those), while teaching it nothing about novel problems. We were building a router over a
growing taxonomy and calling it learning.

---

## 2. SAVE — genuine platonic intelligence (it generalizes)

| Capability | Why it is real (generalizes from structure) | Evidence |
|---|---|---|
| **Numeric homomorphism** | computes value from geometry; no stored facts; works on operands never seen | `84+57=141` exact, untrained operands |
| **Distributional word face** | meaning *emerges* as a superposition of relational context; related cluster, unrelated orthogonal, ambiguity in two basins | gym separation −0.30→+0.125; `bank` near both senses |
| **Composition by reuse** | atoms→word→sentence, digits→number; novel composites cost O(1) | 300 words → 26 atoms |
| **Relaxation / attractor dynamics** (`Reason`) | general Hopfield-style settle: recall, disambiguation-by-context, abstention; works on any clouds | recall 0.96→0.99, disambiguation, abstain-on-unknown |
| **Learned transforms `T(f)`** | few-shot function induction (affine/mult), generalizes to unseen operands | TransformAccumulator / FoldPathDiscovery |
| **The invariants** | frozen identity, G4 conservation, G6 monotonicity, numbers-never-edge | keep the above sound under learning |

These are the things that would still work on a task we never wrote a test for. They are the project.

---

## 3. THROW AWAY / RADICALLY RETHINK — overfitting to a fixed taxonomy

| Thing | Why it is test-fitting | What replaces it |
|---|---|---|
| **PLAN head + 9 hardcoded shapes** | maps ~1:1 to gym skills; the model *classifies* a task into a pre-built template instead of discovering the solution | composition emerges from recognition + the general Fold/Compose primitives, not a shape classifier |
| **Label resolvers** (esp. `ResolvePlanLabel`) | reverse-engineer the task type from output structure to supervise the classifier — "derived not hardcoded" is a fig leaf for encoding the taxonomy | self-supervised prediction; no per-task structural labels |
| **Route head as a 3-way task-classifier** | trained to pick neural/platonic from structural labels; the anchor bug shows how brittle the supervision is | routing **emerges** from substrate confidence (settled→answer, numeric→compute, didn't-settle→abstain) |
| **Pre-built plan routes** (expression-chain, fold-sum/product, predicate, seq *as templates*) | hand-wired for specific gym patterns | keep the *general* primitives (Fold, Compose, Compare-by-sign, Hop); drop the gym-specific pre-wiring |
| **Gym's fixed skill taxonomy as the training TARGET** | shapes the whole system to ~13 synthetic tasks; won't touch anything outside them | train on varied real data by self-supervised prediction; the gym is at most a *probe*, never the target |
| **Neural token decoder as a primary answer path** | the LLM-residue that hallucinates (`big→sluggish`) | the field abstains when it hasn't settled; generation speaks *from* a settled state |

---

## 4. Skeptical of the recent grand work (mine)

- **The "living self" / Creature / consciousness layer.** The *mechanisms* are real (a persistent checkpointed state;
  G6 regeneration). But the tests pass *by construction* — I built `Ablate`+`Regenerate` and then proved regeneration
  works. That demonstrates a mechanism exists; it does **not** demonstrate intelligence. Keep the persistent-self
  vector as a *tool*; treat "consciousness" as motivating narrative, not a claim. Only the relaxation is validated.
- **`FieldCognition` / `FieldOperator`.** Reinvented the route ladder — **throw**. **`Reason` (the relaxation) is the
  keeper** from that work; it is the correct retrieval/disambiguation mechanism and should *be* the retrieval route.
- **The grand vision docs** (`PLATONIC_MIND`, `PLATONIC_CONSCIOUSNESS`). Directionally right (substrate reasons, NN
  thin, alive = self-evidencing) — but only relaxation is proven. Hold the philosophy lightly; this doc is the floor.

---

## 5. The keep-core (the minimal generalizing substrate)

Everything the system needs, and nothing fit to a test:

- **Faces:** numeric (homomorphism) · char (spelling) · word (distributional cloud). Frozen identity, free meaning.
- **General operations (substrate-native, each ABSTAINS when it can't apply):**
  1. **Compute** — the homomorphism (+,−,×,÷, folds). Exact, general.
  2. **Retrieve / disambiguate** — `Reason` relaxation over the clouds: settle → answer; don't-settle → abstain.
  3. **Transform** — learned `T(f)` applied by composition.
  4. **Compose** — atom→word→sentence reuse and the general Fold/Compose (R2), driven by *recognition* of structure.
- **Invariants:** frozen identity restored every write; G4 complement; G6 archive-not-destroy; numbers never edge.

---

## 6. The single substrate-driven control path (replace the task-classifier)

One control flow, **gated by the substrate's own confidence, not a trained task head**:

```
answer(query):
  if query has numeric operands + an operation in context:   # minimal, general op-detection
        return Compute(operands, op)                          # exact homomorphism
  t = Reason(anchors(query))                                  # relax over the clouds
  if t.Settled:        return t.Symbol                        # retrieval / disambiguation
  if a transform/composition is recognised for the context:  # recognition, not a plan classifier
        return Transform/Compose(...)
  return ABSTAIN                                              # the field has nothing to say; no hallucination
```

The selection is **abstention fallthrough by substrate confidence** — which the existing route ladder already does;
what we remove is the **plan/route/op HEADS gating it** and the **structural label resolvers training them**. The
neural layer's *genuine* job shrinks to **conditioning** (learn thresholds, pick among transforms, modulate the
relaxation) — learned by self-supervised prediction, never by classifying gym tasks. `anchors()` must use the
*discriminative* extractor (drop framing-word hubs) — the one place the existing seam is simply buggy.

---

## 7. The training shift

Stop training classifiers on a task taxonomy with structural labels. Train by **self-supervised prediction over
varied data** — predict the continuation the field should settle into — which is simultaneously: how LLMs learn
(so generation stays rich), the free-energy / "alive" principle (minimise surprise), and the thing that makes
capability *emerge* instead of being enumerated. The substrate learns facts by observation (drift-free); the thin
dynamics learn to predict by gradient on a *detached* field (so the two never fight). The gym becomes a held-out
**probe of generalisation**, never the training target.

---

## 8. Concrete keep / throw list

**Keep (the core):** `Core/PlatonicFaceComposer` + `PlatonicFaceDecoder` (homomorphism + codec) · `FaceLayout` ·
`Cognition/Platonic/*` (DialecticalSpace: distributional cloud, composition hubs, `Reason`, invariants) ·
`TransformAccumulator` + `FoldPathDiscovery` · the general substrate routes (compute, geometric/relation retrieval →
folded into `Reason`, learned-function) · the abstention-fallthrough mechanism.

**Throw / shed (the overfit):** the PLAN head + 9 shapes · `ResolvePlanLabel` and the structural label resolvers ·
the route head *as a task-classifier* (keep abstention) · the pre-built plan routes as templates · `FieldCognition`/
`FieldOperator` · the gym as the training target · the neural decoder as a primary path. (Code stays in git history.)

**Fix in place (the one real seam bug):** the trainer's route-label + perception use raw `ExtractMirrorConcepts`
(framing words survive, slot 0 = "a"); make them use the discriminative extractor (`ExtractConceptAnchors` /
`SelectDiscriminativeConcepts`) so a healthy retrieval geometry is actually visible to whatever control path remains.

---

## 9. Honest scope

Validated and real: the homomorphism, the distributional word face, composition-by-reuse, relaxation
(recall/disambiguation/abstention), learned transforms. Everything else is either overfit machinery to shed or
motivating narrative to hold lightly. The work from here is **subtraction toward the generalizing core**, then a
single substrate-driven control path trained by self-supervised prediction — not adding the next classifier.
