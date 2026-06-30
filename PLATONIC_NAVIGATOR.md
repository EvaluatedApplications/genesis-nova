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
a **menu of actions** a single policy chooses among, per step. **The menu is part-built, part north-star** — the
build-state column says which is live today and which is a deliberate future direction we are keeping in view:

| action | what it does | how it produces a target | build-state |
|---|---|---|---|
| **STEP/FOLLOW** (unified) | step to one of the K egocentric relational candidates (a named relation/▷ neighbour = fact recall in one hop) | the candidate's face, landed by the lattice | **BUILT** — the live `QueryNavPolicy` scores exactly these K candidates + HALT and emits the chosen candidate's face as the continuous target |
| **COMPUTE-jump** | apply an operation (e.g. `+`, `category-of`) → a *computed target coordinate* (arithmetic = a target in the void at `141`) | homomorphism / learned functions | **NORTH-STAR (not built)** — the homomorphism-*as-a-walk-step*, a deliberate future direction toward the program-apex (composing math + fact in one walk). Kept in the menu on purpose. |
| **TOWARD-landmark** | move toward a centroid/region | the centroid coordinate | **NORTH-STAR (not built as a distinct action)** — per-level goal-region centroids DO exist (M4 `EnsureLevelRegions`) and steer the walk as a `cand−goal` feature, but there is no separate "emit a centroid target" move yet |
| **HALT** | stop; the current coordinate is the answer | — | **BUILT** — a learned halt head; a confident halt emits the landing, budget exhaustion abstains |

Retrieval is "step to a relational candidate" (BUILT). Arithmetic-as-"compute-jump to the result coordinate" and
composition-as-"step into a structure coordinate" remain the north-star: the live policy chooses among the K
relational candidates + HALT, not yet among compute/landmark moves. There is no privileged ladder — within the BUILT
menu the policy already learns *when* each step applies; widening the menu to COMPUTE-jump is the next frontier, not a
regression to fix.

### 5.1 One motion primitive: every action rides the lattice (BUILT)

The action types above do **not** each get their own traversal machinery. They all reduce to **one** move:
**emit a target coordinate, then let the lattice land the step** — the nearest decodable coordinate to that target, in
O(log N). This is built and live: the policy emits a continuous target face (`NavDecision.Target`) and
`DialecticalSpace.TryLand` resolves **where the foot actually falls** (`NavigatorWalk.Walk`). The BUILT STEP/FOLLOW
move reads the K relational candidates and emits the chosen candidate's face; the north-star COMPUTE-jump /
TOWARD-landmark moves (§5) would *produce a target coordinate* the **same** way and land through the **same**
primitive — that is exactly why widening the menu does not need new traversal machinery. This is the
"rides-on-the-lattice-for-speed" rule: no per-action O(N) scan, no parallel index — the VP-tree (`PlatonicLattice`)
is the single, fast motion primitive for the whole walk.

It also settled the step-granularity question: the policy emits a **continuous** target (a direction/coordinate) and
the **lattice snaps** it to the nearest decodable coordinate — continuous intent, discrete, decodable landing.

### 5.2 The walk grows the space: navigation as genesis tick (BUILT — gated OFF in live walks)

Navigation *can* grow the space, and the write-path is built: `NavigatorWalk` records the face of every passed-through
coordinate, and on a confident successful halt `MaterialiseOnSuccess` commits each one via
`DialecticalSpace.Materialise` — the trail the walk blazed becomes durable structure (the genesis create→select→store
tick, `[[nova-nn-directed-generative-tick]]`), with the existing **relevance-decay eviction** keeping it bounded
(useful coordinates reinforced and kept, dead trails decaying back to latent).

**In production this is OFF.** Every live walk — the gym training rollouts, the held-out eval, the inference
disambiguator hook, the `/nav` REPL probe — uses the default `NavWalkOptions(MaterialiseOnSuccess: false)`: the
navigator currently *reads and answers*, it does not write the store. The genesis-tick growth is realised in code but
left dormant on purpose, so that thinking-creates-structure stays a deliberate, separately-enabled step rather than an
always-on side effect of every query.

---

## 6. The network (thin policy/value over egocentric features) — BUILT

The NN is **thin** — a *recogniser/controller*, never a store (`[[nova-nn-recognizer-space-structural]]`). Built as the
recurrent `NavQueryPolicyNet` + `QueryNavPolicy` driver, conditioned on a **query-context it has without the answer**:

- **Encoder:** embeds the answer-free per-candidate differential rows `[cand−ref, cand−cur, cand, κ]`
  (`NavQueryFeatures.Build`) plus the unified `cand−goal` descent feature (M4), fused with the recurrent self `h_t`
  (seeded from anchor ⊕ cue ⊕ `W_s·self` ⊕ `W_k·kind`, with the cue/kind re-mixed every hop).
- **Policy head (BUILT):** logits over `{the K relational candidates, HALT}` — softmax (sample in training, argmax at
  inference). The **COMPUTE-jump / TOWARD-landmark logits are the north-star (§5), not yet heads**: the live action
  head ranks the K relational candidates and the halt head decides when to stop.
- **Value head (BUILT):** cost-to-go ("am I getting closer"), **MSE-supervised by the oracle `cost[]`** (not an RL
  return) — it is the abstain/over-budget signal, not a policy-gradient critic.
- The NN chooses *which way to step*; the space does all storage/composition/retrieval. It never emits an answer from
  weights — it emits a *position* (a candidate face), which the lattice lands and the substrate decodes.

---

## 7. Training: the flow-field oracle (reverse Dijkstra), then on-policy DAgger — BUILT

> **Build-state correction:** training is **BC warm-start + on-policy DAgger** (`NavQueryDaggerTrainer`,
> `NavDaggerRounds = 2`), with the value head **MSE-supervised by the oracle `cost[]`**. The "Phase 1 — RL fine-tune /
> beat the oracle" below is **NOT built** — it is kept as a north-star (§7 Phase 1). There is no policy-gradient / TD
> return anywhere in the navigator.

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

- **Phase 0 — behavioural cloning on the flow field (BUILT).** For each training `(query, known-answer)`, compute the
  oracle field once (`FlowFieldOracle.Compute`, cached per ancestor); train `π_θ` to reproduce `next[node]` at every
  reachable node (not one path) via masked candidate cross-entropy + halt BCE, while the dense `cost[]` supervises the
  value head via **MSE** (`NavQueryDaggerTrainer.TrainQuery`).
- **Phase 0.5 — on-policy DAgger (BUILT).** Roll the *current* net from each `(member, cue)` with the answer hidden
  (`RolloutQueryTrajectories`, `goalSymbol=null`); the cued flow field still has the correct `next[]` at every strayed
  node, so the walker is taught to **recover from its own slips** — DAgger for free, no teacher re-query.
  `NavDaggerRounds = 2` rounds aggregate with the BC set and retrain each cycle.
- **Phase 1 — RL fine-tune (NORTH-STAR, not built).** Letting the walker **beat** the oracle on the real action menu —
  discovering shorter and compositional routes (math + fact in one walk) the per-goal field can't pre-bake — using
  `cost[]` as a dense reward and frozen-address distance as a fallback heuristic where no field exists, remains a
  future direction. The value head exists as an oracle-supervised cost predictor, not yet as an RL critic. This is the
  same frontier as the COMPUTE-jump action (§5): both are how the walk would *exceed* the taxonomy oracle.
- **Self-supervised surprise (NORTH-STAR):** predicting the next coordinate / whether a step reduces goal-surprise —
  the free-energy framing of `PLATONIC_MIND.md`, *per step* instead of per global settle — is the longer-range
  direction beyond the supervised cost head.

---

## 8. Halt, step budget, and abstention (the cognitive light-cone)

- A **step budget** per query bounds the walker's reach (Levin's cognitive light-cone, `PLATONIC_CONSCIOUSNESS.md`).
- **HALT** is a learned action; halting on a high-confidence decodable coordinate emits the answer.
- **Abstain** falls out structurally: if the budget is exhausted without a confident halt, or the walk lands in
  **undecodable void with no incident edge**, the agent says "I don't know" — the address-space abstention criterion
  (`PLATONIC_NUCLEUS.md` §4), not a tuned threshold.

---

## 9. Substrate wiring (what is built)

**The seams the navigator senses through:**
- `DialecticalSpace.TryDecodeCoordinate(face, …)` — position → (kind, symbol, confidence), realised or latent.
- `DialecticalSpace.Neighborhood(atSymbol, k)` — egocentric neighbours + degree (landmark-ness).
- the decodable frozen address + `FrozenIdentityDistance` (the stable distance the features ride on).
- `_selfField` — the self that initialises `h_t` (read live by the disambiguator hook).

**Built for the navigator (the spec realised):**
- `DialecticalSpace.TryLand(target, …)` — the **land(targetCoord) → coordinate** step: resolves any emitted target to
  its nearest decodable landing. The single motion primitive every action reduces to (§5.1), driven through the
  lattice.
- `DialecticalSpace.Materialise(coord)` — the **materialise write-path** for genesis-tick growth (§5.2). Built and
  callable; **gated OFF in every live walk** (`MaterialiseOnSuccess = false`), so the store is not grown during
  inference today.
- the **flow-field oracle teacher** — `FlowFieldOracle.Compute` / `PlatonicFlowField` (backward Dijkstra → `cost[]` +
  `next[]`), the dense everywhere-defined teacher that replaces a trajectory-recording wrapper around the old routes.
- the **policy/value network + the walk loop** — `NavQueryPolicyNet` + `QueryNavPolicy` + `NavigatorWalk`, the
  navigator proper.
- the **per-level goal-region centroids** — `EnsureLevelRegions` / `NavLevelGoalRegions` derive a landmark face per
  abstraction level from the live graph's depth (M4); they feed the `cand−goal` descent feature. (This is the BUILT
  realisation of "landmark-ness" — degree was the proxy; the goal-region centroid is the cleaner signal. A separate
  TOWARD-landmark *action* remains north-star, §5.)

**North-star seams (not built, kept in view):**
- **COMPUTE-jump** as a target producer (arithmetic + learned-function emit a computed target coordinate the lattice
  lands) — the homomorphism-as-a-walk-step. FOLLOW-edge is already a first-class move (the K relational candidates);
  COMPUTE-jump is the frontier (§5, §7 Phase 1).

---

## 10. Decisions (settled — and how they landed in code)

1. **Action granularity — DECIDED, part-built.** The hybrid menu (STEP / FOLLOW-edge / COMPUTE-jump / TOWARD-landmark /
   HALT) is **unified through the lattice**: every action emits a target coordinate, the lattice lands the step (§5.1).
   **Built today:** STEP/FOLLOW (the K relational candidates) + HALT. **North-star:** COMPUTE-jump / TOWARD-landmark
   (§5). All built moves "ride the lattice for speed."
2. **Step granularity — DECIDED, BUILT (follows from 1).** Policy emits a **continuous** target face; the **lattice
   snaps** it (`TryLand`) to the nearest decodable coordinate. Continuous intent, discrete decodable landing.
3. **Reward / value supervision — LANDED.** The value head is **MSE-supervised by the oracle `cost[]`** (cost-to-go,
   dense, everywhere-defined), not a sparse RL return. Sparse-hit + frozen-address-distance RL shaping was the
   alternative; it was **not** built (the supervised cost head replaced it; RL fine-tune is north-star, §7 Phase 1).
4. **Self — LANDED.** Reuses the engine `_selfField` to seed `h_t` (via `W_s`), threaded across hops; the live
   disambiguator hook reads it so the walk is self-conditioned.
5. **Materialisation on the walk — BUILT, OFF in live.** The write-path exists (`MaterialiseOnSuccess` →
   `DialecticalSpace.Materialise`, genesis tick, `[[nova-nn-directed-generative-tick]]`, bounded by relevance-decay
   eviction), but every production walk runs with it **disabled** (§5.2): the navigator reads/answers, it does not grow
   the store. Navigation *can* write; today it does not.
6. **How much old routing to keep — LANDED differently.** The route ladder was **not** retired. The navigator is wired
   as a **disambiguator hook** in `TryFieldRelax`'s ambiguous branch (§11); the dominant-relation answer above it and
   the one-shot `ds.Reason` below it remain, and a non-confident walk **falls through** to them. The old routing is the
   live fallback, not a teacher scheduled for removal.

---

## 11. Build order (what shipped, in order)

1. **Flow-field oracle — BUILT.** The backward-Dijkstra pattern (`NavMeshFlowField.Compute`, ~30 lines, **not** the
   navmesh geometry) ported over the platonic **action-graph** as `FlowFieldOracle.Compute` / `PlatonicFlowField`:
   from a known answer coordinate it fills `cost[node]` (dense everywhere) + `next[node]` (expert action) over the
   reachable graph, computed once per answer and cached. The reverse-graph is handled (relations reversed, lattice
   steps symmetric). Motion seams landed alongside it: `TryLand` / `Materialise`; the walk senses the frozen address
   via `NavQueryFeatures` / `FrozenIdentityDistance`.
2. **Action seams — PART-BUILT.** FOLLOW-edge is built as the K relational candidates the policy ranks, landed via the
   lattice (§5.1). COMPUTE-jump / TOWARD-landmark *as distinct actions* are north-star (§5); the goal-region landmark
   exists as a `cand−goal` feature (M4), not a move.
3. **Walk loop + thin policy/value net — BUILT.** `NavigatorWalk` + `NavQueryPolicyNet` + `QueryNavPolicy`;
   behavioural cloning on the oracle field (`next[]` everywhere) then on-policy DAgger (`NavDaggerRounds = 2`), with
   `cost[]` supervising the value head by MSE (§7). Trained **every gym cycle** by `TrainNavigatorCycle`, with a
   held-out generalization curve (`RegisterNavigatorHeldOut` / `EvaluateNavigatorHeldOut(PerCue)`).
4. **RL fine-tune — NORTH-STAR (not built).** Beating the oracle on a widened action menu, with frozen-address
   distance as the inference-time heuristic where no field exists, remains the Phase-1 frontier (§7). The value head
   ships as an oracle-cost predictor, not an RL critic.
5. **Live integration — SHIPPED as a HOOK, not a full cut-over.** Rather than `GenerateFromField` *calling* the
   navigator wholesale, `WithProductionMechanisms()` sets `NavigatorDisambiguation = true` and
   `GenesisRuntimeState.WireNavigatorDisambiguator` attaches the trained net to
   `GenesisInferenceEngine.NavigatorDisambiguator` — a delegate consulted **only in the AMBIGUOUS branch** of
   `TryFieldRelax`, *between* the dominant-relation answer and the one-shot `ds.Reason`. The hook walks from the
   subject under the query's learned cue + unified goal-region (self-conditioned), and a **confident, valid, non-self**
   halt is emitted as `navigator-walk` (folding its conclusion into the self); a non-confident / cold / untrained walk
   **falls through** to `ds.Reason` (cold-safe, clear cases never regressed). The old ladder is the live fallback, not
   demoted to a teacher.

> **Reused from NavPathfinder / EvalApp (the genesis-nova lineage):** the flow-field algorithm (step 1). Still
> available as **later** reuse — the `EvalApp` `AdaptiveTuner` (Bayesian/hill-climb self-tuning) to auto-size the
> **step budget** / walk hyperparameters (§8), and the `IStep`/`Pipeline` + `WindowBudgetPressure` framework to host
> the per-hop **tick loop** under a frame/compute budget. Lift the algorithms, not the navmesh domain glue.
