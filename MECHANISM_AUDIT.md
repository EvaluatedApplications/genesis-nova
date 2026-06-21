# Genesis-Nova — Mechanism Audit (blind, code-grounded)

A from-the-code inventory of every mechanism in the platonic space + GRU controller, a conflict
matrix (mechanisms that fight, double-count, or are unwired), and a direct answer to "why don't
things separate in 512 dimensions?". Findings are from reading the actual code, not comments.
Citations are `file:line` at time of audit (2026-06-19).

---

## 0. HEADLINE — why 512 dims don't separate

The 512 dims are largely **illusory for text separation**. Four compounding facts:

1. **Only the word face `[202,512)` (310 dims) is ever measured for text relatedness.**
   `FaceAwareDistance` (`PlatonicSpaceMemory.cs:1733`) uses `RangeDistance(…, 202, dim)` for a text
   query. The **char face `[42,202)` (160 dims) — which holds the text concept's actual identity/
   spelling — is never measured.** Distance ignores exactly the region that carries identity.
2. **That measured word face is filled with ±0.01 seed noise.** `Compose`/`CreateFace` write spelling
   into the *char* face (single tokens) or nothing (multi-word), then `SeedLearnableDims`
   (`PlatonicFaceComposer.cs:293`) puts only ±0.01 RNG noise into `[202,512)`. `AddWordIdentity`
   (`PlatonicFaceComposer.cs:71`) — the one function that writes a distinct near-orthogonal **unit
   code** into that region — **is never called by the creation path (orphaned).** So distinct
   concepts enter the measured region differing only by uniform-magnitude noise. The metric reads the
   *least informative* region of the vector.
3. **Joint unit-normalization discards the radial axis.** `NormaliseFreeRegion`
   (`PlatonicSpaceMemory.cs:1566`) normalizes the whole free region `[0,42)∪[202,512)` to L2=1 as one
   block, after **every** force, and `RestoreFrozenIdentity`+`NormaliseFreeRegion` re-sphere it each
   tick. Everything lives on a unit hypersphere — only direction matters, magnitude is erased, and the
   word-face slice is a sub-block of a jointly-normalized vector (compressed, not itself unit).
4. **So all separation must be manufactured by push/pull from noise, in a compressed angular slice** —
   while the push (live InfoNCE, `InfoNceStep=0.9`) is spiky and the pull (`MessagePassUpdate`) can
   only *rotate* on the sphere, never close the radius gap.

**Highest-leverage fix:** wire `AddWordIdentity` into concept creation so each concept is born with a
distinct near-orthogonal unit code in `[202,512)` — the region the metric actually reads. This inverts
the dynamics from "collapsed-at-noise, fight to separate" to "separated-at-birth, pull related
together." Secondary: reconsider unit-normalization (the radial axis is unusable), and tame
`InfoNceStep`.

---

## 1. Mechanism inventory

### 1A. Face geometry (`Core/FaceLayout.cs`, `Core/PlatonicFaceComposer.cs`)
| region | range (dim=512) | width | frozen for |
|---|---|---|---|
| poly | [0,21) | 21 | numbers (identity) |
| log | [21,42) | 21 | numbers (identity) |
| char | [42,202) | 160 | **text (identity) — but NOT measured by distance** |
| word | [202,512) | 310 | free; **the only region distance reads for text** |

- Number identity = `[0,42)`; free = `[42,512)`. Text identity = `[42,202)`; free = `[0,42)∪[202,512)`.
- `AddWordIdentity` writes a unit whole-string code into `[202,512)` — **orphaned**, never called.
- Spawn-spread births near-orthogonal but from uniform ±0.01 noise in the measured slice.

### 1B. Space dynamics (`Cognition/PlatonicSpaceMemory.cs`) — 7 forces
| # | force | method | cadence | magnitude | drives toward |
|---|---|---|---|---|---|
| F1 | neighbour pull/push | `MessagePassUpdate:1198` | every observation | ~0.04·affinity (floor 0.5) | toward/away neighbour |
| F2 | complement repulsion | `MessagePassUpdate` (dist<1.35) | every obs (conditional) | alpha·0.5 | pos away from ¬pos |
| F3 | contrastive repulsion (manual) | `ApplyContrastiveRepulsionPass:1242` | every 4 obs | 0.014/dist ·10 negs | orthogonality vs unrelated |
| F4 | **contrastive repulsion (InfoNCE) — LIVE default** | same | every 4 obs | 0.9·softmax(−d/0.25) | orthogonality vs nearest confuser |
| F5 | centroid nudge | `ApplyCentroidNudge:1442` | fine-edit only | 0.06/0.03 | toward opposite-side centroid |
| F6 | free-region renormalization | `NormaliseFreeRegion:1566` | after every force | project to ‖·‖=1 | unit sphere |
| F7 | complement enforcement | every method tail | after every force | neg=−pos | G4 |

- Numbers move via F1 attraction; exempt from F3/F4/F5. Pull-then-F6 discards radial component.
- Hubs: attraction is plasticity-damped (`NodePlasticity:1465`, `2/(n+1)`); **repulsion ignores
  plasticity** → hubs easy to push, hard to pull.
- Repulsion exempts 1-hop + 2-hop (shared-neighbour) only → 3-hop-related / post-prune nodes can be
  pushed apart. Negatives are semi-hard (live distance, nearest-first).
- Constants: `MinAttractAffinity=0.5`, `GeometryLearningRate=0.04`, `RepulsionRate=0.02`,
  `RepulsionRatio=0.7`, `RepulsionInterval=4`, `RepulsionSamples=10`, `InfoNceStep=0.9`, `InfoNceTau=0.25`.
  Manual (0.014) and InfoNCE (0.9) are NOT magnitude-matched; reconciled only by softmax Σ=1.

### 1C. Retrieval / distance / eviction
- `FaceAwareDistance:1733`: text query → word-face only; numeric query → `MIN(word, arithFace[0,42))`.
- **Consistent core**: training shapes the word face; semantic VP-tree (`PlatonicLattice`, range
  `[202,512)`) reads the word face.
- Seam 1: the arith-MIN blend measures the **frozen, never-trained** arith face (value-proximity 3↔4).
- Seam 2: `GetNearestConcepts` uses **two metrics** — candidates → `FaceAwareDistance` (blended);
  global → VP-tree semantic-only. Same API, different rankings for numbers.
- Seam 3: VP-tree is a **stale cloned snapshot** (rebuild throttle `max(16,0.05·N)`); bulk retrieval
  reads stale positions. Verification/perception/repulsion dodge this with live faces.
- **Eviction signal = relation-degree (isolation in the *relation graph*) + utility (use/success/
  recency, weights 0.55/0.25/0.20) + obs count. NOT geometric isolation.** Face distance only gates
  duplicate *merge*. A node far in face space but holding edges+use is fully protected.

### 1D. GRU + heads + training (`Model/GenesisNeuralModel.cs`, `Train/GenesisTrainer.cs`)
| head | trained by | loss weight | label/reward |
|---|---|---|---|
| token decode | autograd CE | **1.0** | teacher-forced tokens |
| route (3-way) | autograd CE + perception REINFORCE | 0.25 | `ResolveRouteLabel` (does space retrieve target) |
| query-op (5-way) | autograd CE | 0.25 | `ResolveQueryLabel` (op that fits operands) |
| query-operand | autograd BCE | — | digit-run mask |
| plan (9-way) | autograd CE + perception | 0.25 | `ResolvePlanLabel` (output structure) |
| edit magnitude | **REINFORCE only** | — | within-step retrievability delta |

- GRU is the **shared encoder AND token decoder** (~50M params at 2048) — the heavy lifter, not a thin
  selector. Heads are thin (a few logits each).
- Supervision is **derived from structure** (no answer tables), but label oracles parse symbols
  (`IsExpressionChain`, `TryExtractArithmeticObservation` regex, `ResolveRouteLabel` modality cascade,
  `IsNegativeText="not "`) — walled off from the model, but hardcoded heuristics in the label path.
- Edit reward is a sound within-step causal delta, but **conflates the magnitude's effect with all
  other same-step space writes + noisy geometric reads** → can reward noise.
- REINFORCE heads are correctly isolated (edit/perception step only their own params; shared GRU `.grad`
  protected).
- Rehearsal ring = 64 entries, FIFO, uniform sampling → tail of a large mastered set can still forget.

### 1E. Inference routing + composer (`Infer/GenesisInferenceEngine.cs`, `Cognition/PlatonicGlider.cs`)
- Two-level, **fully GRU-driven**: route head (Mode 0 neural / 1 platonic-direct / 2 assisted) → inside
  Mode 1/2 a fixed cascade: **glider-plan → expression-chain → gru-query → learned-function →
  geometric → relation-first → concept-chain**, each abstaining on failure.
- **No hardcoded inference routing** (the symbol parser was deleted). Composer blocks are a composable
  vocabulary; SEQ scaffold is mined from graded-correct chunks (abstains until learned) — no cue tokens.
- **Router erosion is unmasked**: no a+b short-circuit; a mis-route to Mode 0 loses arithmetic entirely
  (homomorphism stays exact — purely a routing failure).
- Cascade order differs from `CLAUDE.md` (geometric runs before relation-first; CLAUDE.md omits it).

---

## 2. CONFLICT MATRIX (ranked by impact on "won't separate / won't learn")

| # | conflict | mechanisms | effect | fix direction |
|---|---|---|---|---|
| C1 | **Metric reads noise** | `AddWordIdentity` orphaned ↔ `FaceAwareDistance` measures word face | text concepts indistinguishable at birth; separation must be built from noise | **wire `AddWordIdentity` into creation** |
| C2 | **Radial axis unusable** | `NormaliseFreeRegion` re-sphere every tick ↔ attraction / "exile" intent | magnitude erased; attraction can only rotate; no "dark reaches" radius | relax/abandon unit-norm, or use radius as a signal |
| C3 | **Exile→evict not wired** | repulsion (geometry) ↔ eviction (relation-degree+utility) | "stronger repulsion flings junk out for eviction" **does not happen** — geometric isolation isn't an eviction signal | add geometric-isolation OR make junk lose edges (relation-degree drops) |
| C4 | **Spiky push vs gentle/rotating pull** | `InfoNceStep=0.9`, τ=0.25 ↔ attraction 0.04 + rotate-only | oscillation; the recent 0.9 bump worsens it | lower `InfoNceStep` / raise τ / more-frequent-smaller |
| C5 | **Token LM dominates platonic heads 4:1** | token loss 1.0 ↔ route/query/plan 0.25 each, shared GRU | structural cause of router erosion (a+b → neural) | raise platonic weights / decouple encoder / route supervision |
| C6 | **Dual retrieval metric** | `GetNearestConcepts` candidate (blended) ↔ global (semantic-only) | inconsistent neighbours for numbers | unify on one metric |
| C7 | **Blend measures frozen region** | arith-MIN in `FaceAwareDistance` ↔ arith face never trained | numeric queries pulled by value-proximity (3↔4) | drop blend or rely on relation-first |
| C8 | **Hub asymmetry** | attraction plasticity-damped ↔ repulsion ignores plasticity | hubs drift (easy push, hard pull) | apply plasticity to repulsion too |
| C9 | **Edit reward conflation** | edit magnitude credited with all same-step writes + read jitter | noisy credit assignment | isolate the magnitude's own delta |
| C10 | **Route-order surface ambiguity** | glider-plan kind 4 runs before gru-query | same input → "two" or "2" (value-equiv graded, soft) | order by confidence / arbiter |

### Dead code / stale comments / hardcoded heuristics register
- **Orphaned `AddWordIdentity`** (the C1 fix sitting unused).
- **Orphaned Ref shapes** (`PlatonicShapeRegistry` `twicelarger`/`larger`): built, seeded, unit-tested,
  **unreachable in production**; "plan-kind 8" comments are stale (kind 8 = expression-chain).
- **Dead helpers**: `MeanEmbeddingTensor`, `PositionScale` (pre-GRU pooling era).
- **Stale comments**: "gentle ~0.1× attraction" repulsion (actually rebalanced/InfoNCE); perception
  flags "default off" (actually default true); `PredictPlan` doc lists only kinds 0-3 (9 exist).
- **Hardcoded label/observation heuristics** (walled off from model, but present): `IsNegativeText="not "`
  drives edit sign; `TryExtractArithmeticObservation` regex; `ResolveRouteLabel` modality cascade;
  `ObserveArithmeticFaces` fixed 0.05/0.85 contradictions.

---

## 3. Prioritized recommendations
1. **Wire `AddWordIdentity` at concept creation** (C1) — gated by `PushPull_AtScale` separation. Single
   highest-leverage change; directly addresses the dimensionality puzzle.
2. **Revisit unit-normalization** (C2) — the radial axis is dead weight; either use it (radius = a real
   signal, enables "exile") or accept angles-only and stop pretending magnitude is learned.
3. **Decide what "evict junk" means** (C3) — geometric isolation is NOT wired to eviction; either wire
   it or accept that only relation-degree/utility evict.
4. **Tame `InfoNceStep`** (C4) — the 0.9 bump is the top instability suspect; A/B lower values on the
   geometry gate.
5. **Rebalance platonic vs token loss** (C5) — the 4:1 token dominance is the structural router-erosion
   cause.
6. Clean the dead/stale items (low risk, reduces confusion): unify retrieval metric (C6), remove dead
   code, fix stale comments, reach or remove the Ref shapes.
