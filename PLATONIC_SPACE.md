# The Platonic Space — Capabilities Reference

> One place to know what the substrate can do, so designs use it natively instead of bolting structures on top.
> Grounded in the code (file:line where useful). Companion to PROJECT_GLIDER.md.

The platonic space is a **geometric substrate**: every concept is a point (an embedding vector) partitioned
into **faces**, related concepts are positioned by **forces**, and exact computation rides a **homomorphism**
baked into the geometry. Relations, compositions, and functions are **first-class elements in the same space**
— not side tables. Relations are now **positioned relation-elements** (centroid of their endpoints), held in a
keyed index (`_relationIndex`) that plays the same access role `_nodes` plays for concept-elements — an index,
not a graph layered on top. (2026-06-14: promoted off the old `ConceptRelation` side-type.)

---

## 1. The embedding vector and its faces (`Core/FaceLayout.cs`)

A concept's vector of dimension `dim` (production 512) is split into **non-overlapping faces**, each a
functional region. A face is either **identity** (pinned, defines what the concept *is*) or **free**
(learnable "wiggle" — where associations and migrations happen).

| Face | Range | Holds | Encode |
|---|---|---|---|
| **Polynomial (numeric)** | `[0 .. NumericDims)` (≤21) | a number's value, add/sub-homomorphic | `embed[i] = value · 10^-(i+1)` |
| **Logarithmic (numeric)** | `[NumericDims .. 42)` (≤21) | same value, mul/div-homomorphic | `embed[i] = ln|value| · 10^-(i+1)` |
| **Character** | `[42 .. 202)` | a token's spelling, one char per **slot** | per-char atom in slot `i` |
| **Word/chunk** | `[202 .. dim)` | whole words/phrases, one word per **slot** | per-word 5-char-chunk hash |

- **Numeric faces are a homomorphism:** `poly(a)+poly(b)=poly(a+b)`; `log(a)+log(b)=log(a·b)`. Arithmetic is
  therefore **computed in the geometry, exactly**, for any operands — no per-fact storage. (Log face has **no
  representation for 0** — `ln 0` undefined; sign lives in the poly face.)
- **Identity vs free** (`PlatonicFaceComposer.SeedLearnableDims`): for a **number**, identity = the arithmetic
  face `[0..42)` (free = char+word). For a **non-numeric token**, identity = the **char face** `[42..202)`
  (free = numeric `[0..42)` + word). Free dims get tiny deterministic seed noise so they can move. **Key
  consequence:** a word like `"one"` has a FREE numeric face — it *can* migrate onto a value.

---

## 2. Encode / decode — the generative codec (`Core/PlatonicFaceComposer.cs`, `PlatonicFaceDecoder.cs`)

Pure, stateless, exact inverses. This is how a coordinate becomes data and vice-versa.

- **Numeric** (`GetFreshNumericEmbedding` ↔ `DecodeNumericFromPrediction`): value → poly+log faces; decode
  `poly = pred[0]·10`, `log = exp(pred[logStart]·10)`, pick the face with better self-consistency (a
  **quality** ∈[0,1]). **Exact and generalizing.**
- **Char** (`GetCharComposedEmbedding` ↔ `SlotDecodeString`): each char → its hash-atom in slot `i`; decode
  each slot to the nearest of the fixed 95-char vocab, stop at the first near-zero slot. **GENERATIVE** —
  *any* coordinate decodes to *some* string, including unseen ones. This is the "produce unseen inputs" face.
- **Word** (`GetWordComposedEmbedding` ↔ `WordSlotDecode`): each word → a 5-char-chunk-hash atom in a word
  slot. Decode matches each slot to the nearest word in a **supplied vocabulary** → **RECOGNITION, not
  generative** (can't synthesize an unseen word from coordinates). Single tokens live in the *char* face
  (generative); only multi-word strings use the word face.

---

## 3. Elements — the universal currency (`Core/PlatonicTypes.cs`)

Everything in the genesis substrate is a `PlatonicElement` of one `Kind` (enum kept as the conceptual
vocabulary, though only Object/Composition are constructed today):

- **Object** — a concept (vocabulary point).
- **Relation** — a link between elements, positioned at the interpolation/centroid of its endpoints,
  `RelatedTo=[a,b]`. *A relation is itself an element* → it can be related/composed (higher-order). (Conceptual
  today — R1 was removed; relations currently live in `PlatonicSpaceMemory`, see §5/§7.)
- **Composition** — sum of related elements (R2). Produced by `TickExecutor` for the glider blocks.
- **Function** — a learned function applied to an argument. REALIZED (2026-06-14): a function is learned as
  a transform vector `T(f)=avg(embed(out)−embed(in))` (`TransformAccumulator`, no gradient descent) and
  **applied by COMPOSITION** — `embed(x)+T(f)` is a Sum-composition of the argument element with the function
  element, decoded in the function's own face (recorded `PreferredFace`: poly for +k, log for ×k). It is
  **selected from the space by relation** (cue concept → learned relation → function), GRU-route-gated, in the
  learned-operation inference route. Generalizes affine/multiplicative functions to unseen operands from a few
  examples (measured). BINARY learned ops use `FoldPathDiscovery` (a discovered fold — mul = repeated add — or
  log-linear `c=a^α·b^β`) via the same route's two-operand branch. The unary vectors still live in
  `TransformAccumulator` (a positioned store), not yet as `PlatonicElement Kind=Function`.

`PlatonicElement` fields after the 2026-06-14 cleanup: `Id`, `Kind`, `Embedding`, `Symbol`,
`GeneratedAtTick`, `NoveltyScore`, `BridgeConfidence`, `RelatedTo`, `GenerationPath`. (The R6/R7/R9-only
fields — `ComplementId`, `IsHypothesis`, `LocalTransform*`, `IsDevolved`/`Devolved*` — were removed with the
dead tick rules.) `CompositionMode` (**Sum**/Product/Difference/Concatenate) is still used by
`FoldPathDiscovery`/`InputEmbeddingComposer` (only Sum is live).

`PlatonicState` = an immutable `ImmutableArray<PlatonicElement>` + dim + nextId + tick. **One collection holds
all kinds** — objects, relations, compositions, functions together. (This is the unified substrate nova's
`_relations` dict departs from.)

---

## 4. Tick rules — `Core/TickExecutor.cs` (now: R2 Compose only)

`ExecuteTick(action, state) → (state', newElements)`, pure and standalone. These were the genesis "generative
physics" R1–R9 — **hand-coded heuristics**, not learned. **R1 and R3–R9 were removed (2026-06-14):** they ran
only inside the trainer's per-example **tick loop on a private `_tickState` scratchpad** whose elements were
*never* written to the live `PlatonicSpaceMemory` nor read by inference — their sole effect was a telemetry
counter. The fields R6/R7/R9 set on `PlatonicElement` (`LocalTransform*`, `IsDevolved`/`Devolved*`,
`IsHypothesis`, `ComplementId`) were never read and were deleted with them.

Only **R2 Compose** remains — sum the embeddings of an element's related elements into a `Composition`
element. It is **directly consumed** by the glider blocks (`PlatonicGlider.ComposeArithmetic` →
`ExecuteTick(Compose)`) for element-native arithmetic; the meta-layer sum is its oracle.

**Why the rest is gone, not wired:** the substrate is flexible, but R1/R3–R9 ran disconnected and their
capabilities were either already adapted (R2 → glider `Compute`/`Compare`), redundant on the reliable face
(R4 analogy ≈ arithmetic; semantic analogy is lexical/unreliable), or low-value (R3/R7/R9). The north-star way
to add a substrate "verb" is a **glider block** the GRU learns to select/compose — built fresh and tested when
needed (e.g. a future `Fold`/`Function` block) — not a dormant heuristic firing on a scratchpad.

---

## 5. The working memory (`Cognition/PlatonicSpaceMemory.cs`)

The live store + dynamics. Two element collections, each a keyed INDEX into positioned elements (NOT side graphs):
- **Concepts** — `_nodes` of `ConceptNode { PositiveFace, NegativeFace, counts }`.
- **Relations** — `_relationIndex: Dictionary<(left,right), RelationElementNode>`. Each `RelationElementNode`
  is a **positioned relation-element**: its position is the centroid of its endpoints' faces (projected by
  `GetRelationElements`), its strength is `1 − SynthesisContradiction`, plus the learned dynamics/lifecycle.
  Being positioned, a relation can itself be an endpoint of another relation (higher-order). The dict is the
  O(1) access index over these elements — the same role `_nodes` plays for concepts — not a separate graph.

### Forces that position concepts (the geometry IS the relatedness)
- **Observe → update geometry** (`ObserveContradiction` → `UpdateConceptGeometry` → `MessagePassUpdate`): a
  pair's contradiction ∈[0,1] drives **affinity = 1−2·contradiction**; low contradiction → **pull together**,
  high → **push apart**. One canonical message-passing step per observed pair.
- **`MessagePassUpdate`** (the core rule): (1) pull/push the positive face toward/from the neighbour; (2)
  **complement repulsion** — push away from the node's own `NegativeFace` (dual-space separation); (3)
  **restore frozen identity dims** (numbers' arithmetic face, text's char face never drift); (4) **normalize
  the FREE region only** (learned wiggle survives, identity exact); (5) **G4 conservation:**
  `NegativeFace = −PositiveFace` exactly.
- **`ApplyContrastiveRepulsionPass`** (every 16 observations): push each mutable concept away from a few
  sampled UNRELATED concepts → discriminability is *earned* by confirmed attraction. Frozen identity restored.
- **`FineEditFromExample`**: nudge input/output concepts toward their **centroids** (`ComputeCentroid`).
- **`IsFrozenConcept(name) = TryParseNumber`** → numbers are ground truth, never mutated (faces recomputed on
  demand). **Complements:** `¬x` lives at `−embed(x)`; `AreComplements`, hard G4 re-projection.

### Retrieval — `GetNeighbors(concept, type, k, minConf)`, three tiers
- **Relational** — explicit relation edges (lattice adjacency → `_relations` for confidence `1−synthesis`).
- **Numeric** — value-proximity via the numeric lattice ("position IS address"); only for parseable names.
- **Semantic** — face-distance KNN via the VP-Tree, **scoped to the word/semantic face** (face-aware distance).
All merged, ranked by confidence. `GetNearestConcepts` is the bounded, face-aware variant.

### Spatial index (`Cognition/PlatonicLattice.cs`)
Separates discrete topology from continuous geometry: **adjacency** (relation edges, O(1)), **numeric lattice**
(SortedList by parsed value), **char lattice**, **semantic VP-Tree** (KNN on the semantic face, throttled
rebuilds). *Insight: the adjacency already indexes the relation graph — the dict's lookup role is duplicated.*

---

## 6. Invariants & principles (don't violate)

1. **Position is address; relatedness is geometry + relations.** Compare on the *right face* (face-aware), not
   the whole vector.
2. **Numbers relate via the homomorphism, NEVER via relation edges.** Operand↔result / number↔number edges
   pollute and erase prior lessons (measured). Arithmetic = faces; it writes no relation edges. The
   **operation is CLASSIFIED by the GRU op head from context** (`PredictQuery`), not a symbol→op regex — so
   `x` means multiply only when context (flanking operands) says so, and the homomorphism then computes. (The
   hardcoded compact parser was removed 2026-06-14; it forced ambiguous tokens to math.)
3. **Identity faces are frozen; free faces learn.** A migration (e.g. `"one"`→value 1) happens in the free
   numeric face; the char identity stays. (Migration is real but **unstable under repulsion** and the semantic
   face is **lexical** — so geometric retrieval of learned associations was refuted; relations carry what
   geometry can't.)
4. **Equivalence ≠ format.** Value-equivalence (2≡two) vs surface-conversion ("1"→"one" must produce the word)
   — grade accordingly (see AnswerEquivalence).
5. **The substrate is uniform: objects, relations, compositions, functions are all elements.** Prefer adding a
   *kind of element* over a parallel data structure.

---

## 7. Relations as positioned elements (DONE 2026-06-14) — and why the dict stays as an index

The earlier worry was that `_relations` was a relation **graph tacked on top of the space**. Reading the code
showed it is the keyed, **mutable payload + persistence** store (contradiction dynamics + lifecycle counts),
load-bearing for `ObserveContradiction`/`ReinforceEvidence` (the retention-critical signal), snapshots, and
pruning. The `PlatonicState`/`PlatonicElement` collection is immutable + linear-scan (for the tick/glider
substrate) — unsuited to per-observation keyed mutation. So a literal "store relations in `PlatonicState` and
delete the dict" would be O(N) per observation, break persistence, and risk retention — the wrong engineering.

What was the *real* defect was conceptual: relations weren't positioned, and the dict read like a side-graph.
Fixed by **promoting** them:
- `ConceptRelation` → **`RelationElementNode`**, a first-class **positioned element** — position = centroid of
  its endpoints' faces (derived, always consistent; projected by `GetRelationElements`), strength = `Strength`
  (`1 − SynthesisContradiction`). Being positioned, a relation can itself be a relation endpoint (higher-order).
- `_relations` → **`_relationIndex`**: documented as the O(1) **access index** over relation-elements — the
  same role `_nodes` plays for concept-elements — **not** a graph on top. (Every spatial store needs an index.)
- Retrieval (the Relational tier + `TryRelationElementNeighbour`) is the **canonical element-native traversal**,
  reading each element's `Strength`. Behavior-preserving; retention held (full suite green).

Net: relations are now positioned, composable elements; the dict is honestly an index, not a parallel graph.
