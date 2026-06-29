# The Navigator: reasoning as a learned walk through the address space

> **Status: DESIGN / PROPOSAL.** The substrate it rides on is built and proven (the decodable address space —
> `PLATONIC_NUCLEUS.md`); the navigator itself is not built yet. This doc is the spec to shape and build against.
> Open decisions are collected in §10 — they are yours to settle.

> Reasoning is **not** a one-shot route to an endpoint, and it is **not** the whole field relaxing to a global
> equilibrium. Reasoning is a **situated walk**: a neural agent stands *at a coordinate*, senses its local
> neighbourhood in first person, and **steps** — hop by hop, choosing its own path — until it stands on the answer or
> halts. The answer is **where the walk ends**, not what a fixed ladder retrieves.
>
> This reframes `PLATONIC_MIND.md` §3 ("reasoning is relaxation") rather than discarding it. The free-energy /
> surprise principle is kept, but **re-located**: instead of the *entire* field settling at once (global,
> uncontrolled, prone to chaotic hub-dilution), a *situated agent* descends a **local** surprise gradient one step at
> a time, and the **policy** — not the dynamics — picks the direction. Relaxation becomes one **sensor** the walker
> can consult (and override), not the decision-maker. That is what attenuates the chaos: navigation imposes a
> trajectory the field's equilibrium-seeking never had to.
>
> Companions: `PLATONIC_NUCLEUS.md` (the decodable address space the walker moves through), `PLATONIC_CONSCIOUSNESS.md`
> (the self that does the moving), `PLATONIC_THEORY.md` (Law A — coordinates are elements, realised or latent).

---

## 1. Why the address space makes this possible

A space is only *walkable* if its local structure is stable and decodable. The address-space refactor is exactly that
precondition (`PLATONIC_NUCLEUS.md`):

- **Every coordinate decodes** (`TryDecodeCoordinate`) — realised *or* latent — so the walker always knows "where am
  I", and **latent coordinates are reachable waypoints** (it can step toward `141` or a never-stored word). The void
  is navigable terrain, not fog.
- **Frozen bands don't drift**, so the neighbourhood is stable from step to step (on the old learned clouds, the
  ground would shift under each footfall — the chaos we are attenuating).
- **Distance in the frozen address is a dense, stable training signal** (how much closer did that step get me to the
  goal?) — only definable because identity is decodable. This is what makes the walk *learnable* rather than a sparse
  needle-in-a-haystack search.

The navigator is the control layer that the route ladder used to be — but it *learns* the space at the atomic level
by moving through it, instead of classifying a query into a fixed taxonomy of routes.

---

## 2. The agent and its world (the decision process)

A walk is a sequence of `(state → observe → act)` until halt. Framed as an MDP whose *environment is the substrate*:

```mermaid
flowchart LR
  S["state s_t<br/>position p_t · self h_t · goal g"] --> O["observe o_t<br/>(first person, egocentric)"]
  O --> A["policy π_θ<br/>choose action a_t"]
  A --> T["step → p_{t+1}<br/>(substrate is the environment)"]
  T --> S
  A -.HALT.-> E["decode p_t → answer"]
  classDef n fill:#1a5276,color:#fff,stroke:#85c1e9; class S,O,A,T n;
  classDef h fill:#922b21,color:#fff,stroke:#f1948a; class E h;
```

- **State `s_t` = (position `p_t`, self `h_t`, goal `g`).** `p_t` is a *coordinate* (not necessarily a stored node).
  `g` is the encoded query/question. `h_t` is the self-object (§3).
- **Observation `o_t` (first person — what the agent senses from where it stands):**
  - the **decoded identity** of `p_t` — `TryDecodeCoordinate(p_t)` → (kind, symbol, confidence);
  - the **egocentric neighbourhood** — `Neighborhood(p_t, k)` → the k nearest coordinates, each with its *relative
    bearing* (Δ in frozen-address space), distance, decoded identity, relation-degree (landmark-ness), and edge-type
    if a relation links them;
  - **landmark bearings** — direction/distance to relevant **centroids** (category attractors), so the walker can
    head toward a region, not just a point;
  - the **goal-relative signal** — how each neighbour aligns with `g` (does stepping there reduce frozen-address
    distance to where the answer should be?);
  - the **relaxation hint** (subordinated) — the local free-energy gradient at `p_t`: which way the field *would*
    settle. One sensor among many; the policy may follow or ignore it.
- **Action `a_t` (§5).** Move to a neighbour, take a *compute/relation jump*, or HALT.
- **Transition.** Deterministic: the chosen action sets `p_{t+1}`. No environment stochasticity — the space is fixed
  during a walk.
- **Reward (training only, §7).** Reach the answer coordinate in few hops; penalise wandering, dead-ends, and
  exceeding the step budget.

---

## 3. The self-object: first-person continuity

`h_t` is a **recurrent hidden state carried across hops** — the agent's working memory and point of view as it moves.
It is what makes the walk *first person*: one self threads the whole trajectory, accumulating where it has been and
what it is looking for.

- **Initialised** from the query `g` and the persistent **`_selfField`** (the meaning-space self that already
  conditions cognition — `[[nova-self-conditions-cognition]]`). The walk *starts as* the self, oriented by the
  question.
- **Updated each hop:** `h_{t+1} = f_θ(h_t, o_t, a_t)` — it integrates what it just saw and did.
- **Conditions the policy:** the same `h_t` that remembers the path also biases the next step, so the self is
  *load-bearing*, not decorative — it literally drives the trajectory.

This is the concrete realisation of "a self that navigates the space in first person": the self is not a label on the
side of cognition; it is the moving frame cognition happens in.

---

## 4. How this attenuates the chaos

Global relaxation lets the *entire* field grind toward equilibrium — an uncontrolled dynamical system, the source of
instability and hub-dilution (a populous category's basin swallows everything near it). The walker replaces that with
a **controlled trajectory**:

- only the **local** neighbourhood is consulted each step — no global settle is ever required;
- the **policy** decides the direction, so it can *route around* a dilution hub instead of being sucked in;
- **relaxation is demoted** to a local hint in `o_t` (and a teacher, §7) — useful gradient, not verdict.

So the field never has to reach equilibrium for the mind to think; the NN **samples** the space along a goal-directed
path. The chaos is attenuated because order is imposed by navigation, not awaited from the dynamics.

---

## 5. The action space — the old routes become actions

The route ladder's fixed sequence (op → composer → function → relaxation → retrieval → chain → neural) collapses into
a **menu of actions** a single policy chooses among, per step:

| action | what it does | substrate primitive |
|---|---|---|
| **STEP-near** | move to one of the k egocentric neighbours | `Neighborhood` |
| **FOLLOW-edge** | step along a named relation/▷ from `p_t` (fact recall = one hop) | adjacency / relations |
| **COMPUTE-jump** | apply an operation (e.g. `+`, `category-of`) → a *computed target coordinate*, step there (arithmetic = route into the void to `141`) | homomorphism / learned functions |
| **TOWARD-landmark** | move toward a centroid/region bearing | centroids |
| **HALT/EMIT** | decode `p_t` as the answer and stop | `TryDecodeCoordinate` |

Retrieval is "follow an edge." Arithmetic is "compute-jump to the result coordinate." Composition is "step into a
structure coordinate." There is no privileged ladder — the policy learns *when* each move applies. (Recommendation:
include COMPUTE-jump and FOLLOW-edge as first-class actions; a pure spatial walk cannot do arithmetic or one-hop
recall — see §10.)

---

## 6. The network (thin policy/value over egocentric features)

Keep the NN **thin** — a *recogniser/controller*, never a store (`[[nova-nn-recognizer-space-structural]]`):

- **Encoder:** embed `o_t` — neighbour faces + decoded identities + bearings + goal-alignment + relaxation hint —
  fused with the self `h_t`.
- **Policy head:** logits over `{the k STEP-near targets, FOLLOW-edge options, COMPUTE-jump ops, TOWARD-landmark,
  HALT}`; softmax (sample in training, argmax at inference).
- **Value head:** expected return / "am I getting closer" (for RL and for an abstain signal).
- The NN chooses *which way to step*; the space does all storage/composition/retrieval. It never emits an answer from
  weights — it emits a *position*, which the substrate decodes.

---

## 7. Training: imitation first, then reinforcement

The space is enormous; a from-scratch random walk never hits the answer. Cold-start by **imitation**, then improve.

- **Phase 0 — behavioural cloning from teacher trajectories.** Today's heuristic routes + relaxation already answer
  many training queries and *touch a path* doing it (query concept → relation neighbour → answer). Convert those into
  target trajectories; train `π_θ` to reproduce the teacher's next step. (This is why we did **not** delete the old
  routing — it is the teacher.)
- **Phase 1 — DAgger / RL fine-tune.** Reward reaching the answer in few hops; when the walker strays, query the
  teacher for the correct next step (DAgger) and add it to the data. Let it then **beat** the teacher — discover
  shorter paths and compositional routes (math + fact in one walk) the fixed ladder never could.
- **Dense shaping from the address space:** reward each step that *decreases frozen-address distance to the target
  coordinate*. This dense signal — the thing a learned walk needs — exists **only because identity is decodable**
  (§1). Sparse "hit/miss" alone would not train.
- **Self-supervised surprise:** predict the next coordinate / predict whether a step reduces goal-surprise — the
  free-energy framing of `PLATONIC_MIND.md`, now *per step* instead of per global settle.

---

## 8. Halt, step budget, and abstention (the cognitive light-cone)

- A **step budget** per query bounds the walker's reach (Levin's cognitive light-cone, `PLATONIC_CONSCIOUSNESS.md`).
- **HALT** is a learned action; halting on a high-confidence decodable coordinate emits the answer.
- **Abstain** falls out structurally: if the budget is exhausted without a confident halt, or the walk lands in
  **undecodable void with no incident edge**, the agent says "I don't know" — the address-space abstention criterion
  (`PLATONIC_NUCLEUS.md` §4), not a tuned threshold.

---

## 9. Substrate wiring (what exists, what the navigator still needs)

**Already built (the seams):**
- `DialecticalSpace.TryDecodeCoordinate(face, …)` — position → (kind, symbol, confidence), realised or latent.
- `DialecticalSpace.Neighborhood(atSymbol, k)` — egocentric neighbours + degree (landmark-ness).
- the decodable frozen address + `FrozenIdentityDistance` (the stable distance for reward shaping).
- `_selfField` — the self to initialise `h_t` from.

**Still needed for the navigator (later build):**
- a **centroid/landmark accessor** (TOWARD-landmark bearings) — degree is a proxy today; a real centroid index is
  cleaner.
- **COMPUTE-jump / FOLLOW-edge** exposed as steppable actions returning a target coordinate (arithmetic +
  learned-function + relation traversal), so the homomorphism and recall are first-class moves.
- a **trajectory-recording teacher** wrapper around the current routes (Phase 0 data).
- the **policy/value network + the walk loop** (the navigator proper), replacing `GenerateFromField`'s route ladder.

---

## 10. Open decisions (yours to settle)

1. **Action granularity.** Pure neighbour-stepping, or also COMPUTE-jump + FOLLOW-edge + TOWARD-landmark?
   *(Recommend: include them — else no arithmetic / no one-hop recall.)*
2. **Discrete vs continuous step.** Choose among discrete `Neighborhood` candidates (simple, stable), or emit a
   continuous direction vector and snap to the nearest decodable coordinate (expressive, needs snapping)?
   *(Recommend: discrete first, continuous later.)*
3. **Reward.** Sparse hit + dense frozen-address-distance shaping — confirm, or sparse-only?
   *(Recommend: shaped — the space is too big for sparse.)*
4. **Self.** Reuse/extend `_selfField` as `h_t`, or a fresh recurrent state seeded from it? *(Recommend: reuse +
   extend.)*
5. **Materialisation on the walk.** Does passing through a latent coordinate ever *realise* it (leave a trail, grow
   the space — the genesis tick, `[[nova-nn-directed-generative-tick]]`), or is navigation read-only? *(Open — this is
   how the space could grow from thinking.)*
6. **How much old routing to keep.** Keep all routes as the teacher through Phase 1, then retire — or keep some as
   permanent fallbacks?

---

## 11. Build order (once §10 is settled)

1. **Teacher trace** — wrap the current routes to emit `(query, trajectory, answer)` records.
2. **Action seams** — COMPUTE-jump / FOLLOW-edge / TOWARD-landmark + centroid accessor on the substrate.
3. **Walk loop + policy/value net** — the navigator; behavioural cloning on the teacher traces (Phase 0).
4. **DAgger/RL fine-tune** — dense address-distance reward; let it beat the teacher (Phase 1).
5. **Cut over** — `GenerateFromField` calls the navigator; old ladder demoted to teacher/fallback; re-earn the
   skipped routing tests (`MeaningTick`, `FringeAssociation`) as *navigation* outcomes.
