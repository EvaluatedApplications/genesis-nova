# The Navigator: reasoning as a learned walk through the address space

> **Status: BUILT & LIVE (M1–M4 + multi-hop ceiling-break).** This started as a design proposal; the navigator it
> specs is now built, trained in the gym every cycle, and **ON in production** —
> `GenesisNovaConfig.WithProductionMechanisms()` sets `NavigatorDisambiguation = true`, which wires the trained policy
> into the **AMBIGUOUS branch** of `GenesisInferenceEngine.TryFieldRelax` (between the dominant-relation answer and the
> one-shot `ds.Reason`), gated to a confident halt so a cold/untrained walk falls through (cold-safe). Code:
> `src/GenesisNova/Cognition/Navigator/` — `PlatonicFlowField.cs` (the backward-Dijkstra flow-field oracle/teacher),
> `NavQueryFeatures.cs` + `NavQueryPolicyNet.cs` (the **answer-free, query-conditioned** recurrent policy/value net +
> `QueryNavPolicy`), `NavQueryDaggerTrainer.cs` (BC warm-start + on-policy DAgger), `NavigatorWalk.cs` (the walk loop +
> `FlowFieldPolicy` oracle seam); live wiring in `Runtime/GenesisRuntimeState.WireNavigatorDisambiguator`; per-cycle
> training, held-out eval and learned level-regions in `Runtime/GenesisEvalAppRuntime.Navigator.cs`.
>
> **Read the body below as the originating vision — several pieces were realised differently than specced (notes inline,
> and the design's "open decisions" in §10 are settled in code):**
> - The walk is conditioned on a **QUERY-CONTEXT it has without the answer** — the anchor concept + a **learned level
>   cue** over `{GENUS, DOMAIN, ROOT}` (the `NavCue` enum; anchors `∘gns`/`∘dom`/`∘rut`, the level read from the answer's
>   **graph depth** via `LearnNavLevelCue`/`DeriveNavCue`, no word list) — **not** a supplied goal coordinate `g` (§2).
>   The candidate rows are answer-free differentials `[cand−ref, cand−cur, cand, κ]` (`NavFeatures`/`NavQueryFeatures`).
> - Training is **BC-on-the-flow-field then on-policy DAgger** (the §7 "Phase 1 — RL fine-tune / beat the oracle" was
>   **not** built; the value head is supervised by the oracle `cost[]` via MSE, not a policy-gradient return).
> - **M2** adds target-kind conditioning (`DeriveNavKind` → `W_k·kindFace`, halt-on-first-of-this-kind composition).
> - **M4** adds a **learned per-level goal-region** centroid (`EnsureLevelRegions`/`NavLevelGoalRegions`); the unified
>   `cand−goal` feature + `W_k` halt-bias is what broke the multi-hop (DOMAIN/ROOT) landing ceiling (~0%→100% on a clean
>   taxonomy). Held-out generalization is measured by `RegisterNavigatorHeldOut`/`EvaluateNavigatorHeldOut(PerCue)`.
> - **§5.2 materialise-on-walk** exists (`NavigatorWalk.MaterialiseOnSuccess` + `DialecticalSpace.Materialise`) but is
>   **off** in every live walk (the navigator currently reads/answers; it does not grow the store).

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

| action | what it does | how it produces a target |
|---|---|---|
| **STEP-near** | move to one of the k egocentric neighbours | the lattice neighbours themselves |
| **FOLLOW-edge** | step along a named relation/▷ from `p_t` (fact recall = one hop) | the edge's endpoint coordinate |
| **COMPUTE-jump** | apply an operation (e.g. `+`, `category-of`) → a *computed target coordinate* (arithmetic = a target in the void at `141`) | homomorphism / learned functions |
| **TOWARD-landmark** | move toward a centroid/region | the centroid coordinate |

Retrieval is "follow an edge." Arithmetic is "compute-jump to the result coordinate." Composition is "step into a
structure coordinate." There is no privileged ladder — the policy learns *when* each move applies.

### 5.1 One motion primitive: every action rides the lattice (DECIDED)

The action types above do **not** each get their own traversal machinery. They all reduce to **one** move:
**emit/compute a target coordinate, then let the lattice land the step** — the nearest decodable coordinate to that
target, in O(log N). STEP-near reads lattice neighbours directly; FOLLOW-edge, COMPUTE-jump, and TOWARD-landmark each
*produce a target coordinate* and the **lattice resolves where the foot actually falls**. This is the
"rides-on-the-lattice-for-speed" rule: no per-action O(N) scan, no parallel index — the VP-tree (`PlatonicLattice`)
is the single, fast motion primitive for the whole walk.

It also settles the step-granularity question: the policy emits a **continuous** target (a direction/coordinate) and
the **lattice snaps** it to the nearest decodable coordinate — continuous intent, discrete, decodable landing. The
one requirement this puts on the substrate (§9): the lattice must index/query the **frozen address bands** (stable
identity), not the drifting orbital tail, so a "land near coordinate X" query is exact and drift-free.

### 5.2 The walk grows the space: navigation as genesis tick (DECIDED)

Navigation is **not** read-only. When the walker passes through a **latent** coordinate that proves *useful* — it lies
on a successful path / has high goal-alignment / the walk halts confidently there — that coordinate is
**materialised**: committed to the store with an orbital, so the trail the walk blazed becomes durable structure.
Thinking *creates* structure (the genesis create→select→store tick, `[[nova-nn-directed-generative-tick]]`). The
existing **relevance-decay eviction** keeps this bounded — useful materialised coordinates are reinforced and kept,
trails that never pay off decay back to latent. The store therefore converges on *exactly the useful* structure:
navigation writes, eviction prunes. (A write-policy — *when* a passed-through coordinate earns materialisation — is a
build detail in §11; default: materialise on a confident successful halt-path, let decay handle the rest.)

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

## 7. Training: the flow-field oracle (reverse Dijkstra), then reinforcement

The space is enormous; a from-scratch random walk never hits the answer. Cold-start by **imitating an oracle** — and
the right oracle is **not** A\* (one path) and **not** the degraded old routes (a weak teacher). It is a **backward
Dijkstra from the answer** — a *flow field* — adapted from the proven NavPathfinder primitive
(`NavMeshFlowField.Compute` → `(cost[], next[])`; we lift the ~30-line algorithm, not the navmesh geometry).

**The oracle = one backward Dijkstra from the answer coordinate over the action-graph.** Nodes are
concepts/coordinates; edges are the action menu reversed (FOLLOW-edge → relations reversed `category ← entity`;
STEP-near → symmetric lattice neighbours; COMPUTE-jump → reversible ops, handled specially). It fills:
- **`cost[node]`** — the exact optimal cost-to-answer from *every* node → the **dense reward field**, defined
  everywhere (not just along one path);
- **`next[node]`** — the optimal next action from *every* node → the **expert policy, everywhere**.

Computed **once per answer**, cached, shared across every query to that answer (the NavPathfinder amortisation).

Why a flow field beats A\* *as a teacher*: A\* labels one trajectory; the flow field labels the **whole reachable
graph** with the expert action. So when the learner strays off the optimal path, the field **already** has the
correct next move there — **DAgger for free**, no teacher re-query — and `cost[]` gives a dense distance-to-goal
reward at every state the learner could occupy.

- **Phase 0 — behavioural cloning on the flow field.** For each training `(query, known-answer)`, compute the
  oracle field once; train `π_θ` to reproduce `next[node]` at every reachable node (not one path). The dense
  `cost[]`-gradient supervises the value head.
- **Phase 1 — RL fine-tune.** Let the walker **beat** the oracle on the real action menu — discover shorter and
  compositional routes (math + fact in one walk) the per-goal field can't pre-bake — using `cost[]` as the dense
  reward and the frozen-address distance as a fallback heuristic where no field exists (inference, answer unknown).
- **Self-supervised surprise:** predict the next coordinate / whether a step reduces goal-surprise — the free-energy
  framing of `PLATONIC_MIND.md`, now *per step* instead of per global settle.

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
- the **lattice over the FROZEN ADDRESS** — the single motion primitive (§5.1). The VP-tree must index/query the
  stable frozen bands `[42,416)`, not the drifting orbital tail, and expose a "nearest decodable coordinate to target
  `X`" query. (This also fixes the L2-flagged stale slice: the lattice still keys on `WordFaceStart=202` while
  semantics moved to `416`.)
- a **land(targetCoord) → coordinate** step: the lattice resolves any emitted/computed target to its nearest
  decodable landing — the move every action reduces to.
- a **centroid/landmark accessor** (TOWARD-landmark targets) — degree is a proxy today; a real centroid index is
  cleaner.
- **COMPUTE-jump / FOLLOW-edge** as *target producers* (arithmetic + learned-function + relation traversal emit a
  target coordinate; the lattice lands it), so the homomorphism and recall are first-class moves.
- a **materialise(coord) write-path** for the genesis-tick growth (§5.2) — commit a useful passed-through latent
  coordinate to the store; reuse the existing relevance-decay eviction to bound it.
- a **trajectory-recording teacher** wrapper around the current routes (Phase 0 data).
- the **policy/value network + the walk loop** (the navigator proper), replacing `GenerateFromField`'s route ladder.

---

## 10. Decisions (settled)

1. **Action granularity — DECIDED.** Hybrid menu (STEP / FOLLOW-edge / COMPUTE-jump / TOWARD-landmark / HALT),
   **unified through the lattice**: every action emits a target coordinate, the lattice lands the step (§5.1). All
   moves "ride the lattice for speed."
2. **Step granularity — DECIDED (follows from 1).** Policy emits a **continuous** target; the **lattice snaps** it to
   the nearest decodable coordinate. Continuous intent, discrete decodable landing.
3. **Reward — default taken.** Sparse hit + dense frozen-address-distance shaping. *(Override if you want sparse-only.)*
4. **Self — default taken.** Reuse + extend `_selfField` as `h_t`. *(Override if you want a fresh recurrent state.)*
5. **Materialisation on the walk — DECIDED.** The walk **grows the space** (§5.2): useful passed-through latent
   coordinates are materialised (genesis tick, `[[nova-nn-directed-generative-tick]]`); relevance-decay eviction
   bounds it. Navigation is **not** read-only.
6. **How much old routing to keep — default taken.** Keep all routes as the teacher through Phase 1, then retire.
   *(Override if you want permanent fallbacks.)*

---

## 11. Build order (§10 settled)

1. **Flow-field oracle** *(done partly — motion seams landed: `TryLand`/`Materialise`; the navigator senses the frozen address via `NavQueryFeatures` / `FrozenIdentityDistance`)*.
   Port the backward-Dijkstra pattern (`NavMeshFlowField.Compute`, ~30 lines, **not** the navmesh geometry) over the
   platonic **action-graph**: from a known answer coordinate, fill `cost[node]` (dense reward) + `next[node]` (expert
   action) over the reachable graph. Compute once per answer, cache. Handle the reverse-graph (relations reversed,
   lattice steps symmetric, compute-jumps special).
2. **Action seams** — COMPUTE-jump / FOLLOW-edge / TOWARD-landmark as target-producers (landing via the lattice, §5.1).
3. **Walk loop + thin policy/value net** — behavioural cloning on the oracle field (`next[]` everywhere; `cost[]`
   supervises the value head). Phase 0.
4. **RL fine-tune** — dense `cost[]` reward; let it beat the oracle on the live action menu; frozen-address distance
   as the inference-time heuristic where no field exists. Phase 1.
5. **Cut over** — `GenerateFromField` calls the navigator; old ladder demoted to fallback; re-earn the skipped routing
   tests (`MeaningTick`, `FringeAssociation`) as *navigation* outcomes.

> **Reusable from NavPathfinder / EvalApp (the genesis-nova lineage):** the flow-field algorithm (step 1, primary);
> and as **secondary, later** reuse — the `EvalApp` `AdaptiveTuner` (Bayesian/hill-climb self-tuning) to auto-size the
> **step budget** / walk hyperparameters (§8), and the `IStep`/`Pipeline` + `WindowBudgetPressure` framework to host
> the per-hop **tick loop** under a frame/compute budget. Lift the algorithms, not the navmesh domain glue.
