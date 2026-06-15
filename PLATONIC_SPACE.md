# The Platonic Space — Capabilities Reference

> One place to know what the substrate can do, so designs use it natively instead of bolting structures on top.
> Grounded in the code (file:line where useful). Companion to PROJECT_GLIDER.md.

The platonic space is a **geometric substrate**: every concept is a point (an embedding vector) partitioned
into **faces**, related concepts are positioned by **forces**, and exact computation rides a **homomorphism**
baked into the geometry. Relations, compositions, and functions are **first-class elements in the same space**
— not side tables. Relations are **positioned relation-elements** (centroid of their endpoints), held in a
keyed index (`_relationIndex`) that plays the same access role `_nodes` plays for concept-elements — an index,
not a graph layered on top.

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
  therefore **computed in the geometry, exactly**, for any operands — no per-fact storage. (The log face has
  **no representation for 0** — `ln 0` undefined; sign lives in the poly face.)
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

Everything in the substrate is a `PlatonicElement` of one `Kind`:

- **Object** — a concept (vocabulary point).
- **Relation** — a link between elements, positioned at the centroid of its endpoints, `RelatedTo=[a,b]`.
  *A relation is itself an element* → it can be related/composed (higher-order). Relations live in
  `PlatonicSpaceMemory` as positioned relation-elements (see §5/§7).
- **Composition** — sum of related elements (R2). Produced by `TickExecutor` for the glider blocks.
- **Function** — a function as a first-class element. Two complementary forms:
  - **Learned transforms** — a function learned as a transform vector `T(f)=avg(embed(out)−embed(in))`
    (`TransformAccumulator`, no gradient descent) and **applied by COMPOSITION** (`embed(x)+T(f)` is a
    Sum-composition of the argument with the function, decoded in the function's `PreferredFace`: poly for +k,
    log for ×k). Selected from the space by relation (cue concept → learned relation → function),
    GRU-route-gated, in the learned-operation route. Generalizes affine/multiplicative functions to unseen
    operands from a few examples. Binary learned ops use `FoldPathDiscovery` (a discovered fold — mul =
    repeated add — or log-linear `c=a^α·b^β`).
  - **Shape elements** — composer shapes registered as positioned `ElementKind.Function` elements
    (`RegisterFunctionElement` / `FunctionElements`), each with a `RelatedTo` pointing at the shape-elements it
    composes (a Ref shape references its sub-shapes). Held in `_functionElements`, parallel to `_nodes` and
    `_relationIndex`. `PlatonicShapeRegistry` holds the executable glider definitions and materializes these
    elements; the interpreter executes a `Ref` by traversing them recursively.

`PlatonicElement` fields: `Id`, `Kind`, `Embedding`, `Symbol`, `GeneratedAtTick`, `NoveltyScore`,
`BridgeConfidence`, `RelatedTo`, `GenerationPath`. `CompositionMode` (**Sum**/Product/Difference/**Concatenate**)
is used by `FoldPathDiscovery`/`InputEmbeddingComposer` (Sum live) and names the Seq shape's binding
(Concatenate).

`PlatonicState` = an immutable `ImmutableArray<PlatonicElement>` + dim + nextId + tick, used by the tick/glider
substrate. **One collection holds all kinds** — objects, relations, compositions, functions together.

---

## 4. Tick rules — `Core/TickExecutor.cs` (R2 Compose)

`ExecuteTick(action, state) → (state', newElements)`, pure and standalone. Only **R2 Compose** is live — sum
the embeddings of an element's related elements into a `Composition` element. It is **directly consumed** by
the glider blocks (`PlatonicGlider.ComposeArithmetic` → `ExecuteTick(Compose)`) for element-native arithmetic
and the N-way Fold; the meta-layer sum is its oracle.

The way to add a substrate "verb" is a **glider block** the GRU learns to select/compose — built fresh and
tested when needed — not a dormant heuristic firing on a scratchpad.

---

## 5. The working memory (`Cognition/PlatonicSpaceMemory.cs`)

The live store + dynamics. Element collections, each a keyed INDEX into positioned elements (NOT side graphs):
- **Concepts** — `_nodes` of `ConceptNode { PositiveFace, NegativeFace, counts }`.
- **Relations** — `_relationIndex: Dictionary<(left,right), RelationElementNode>`. Each `RelationElementNode`
  is a **positioned relation-element**: its position is the centroid of its endpoints' faces (projected by
  `GetRelationElements`), its strength is `1 − SynthesisContradiction`, plus the learned dynamics/lifecycle.
  Being positioned, a relation can itself be an endpoint of another relation (higher-order). The dict is the
  O(1) access index over these elements — the same role `_nodes` plays for concepts — not a separate graph.
- **Function elements** — `_functionElements` (shapes-as-elements; see §3).
- **Chunk-element store** — `_chunkStore`: text chunks mined from graded-correct outputs, grouped by tag
  (`MineChunk` / `TryGetTopChunk`), the cache the Seq shape binds. Persisted in `PlatonicMemorySnapshot`;
  restored additively so a maintenance pass never wipes it.

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
  demand). **Complements:** `¬x` lives at `−embed(x)`; hard G4 re-projection.

### Retrieval — `GetNeighbors(concept, type, k, minConf)`, three tiers
- **Relational** — explicit relation edges (lattice adjacency → `_relationIndex` for confidence `1−synthesis`).
- **Numeric** — value-proximity via the numeric lattice ("position IS address"); only for parseable names.
- **Semantic** — face-distance KNN via the VP-Tree, **scoped to the word/semantic face** (face-aware distance).
All merged, ranked by confidence. `GetNearestConcepts` is the bounded, face-aware variant.

### Spatial index (`Cognition/PlatonicLattice.cs`)
Separates discrete topology from continuous geometry: **adjacency** (relation edges, O(1)), **numeric lattice**
(SortedList by parsed value), **char lattice**, **semantic VP-Tree** (KNN on the semantic face, throttled
rebuilds).

---

## 6. Invariants & principles (don't violate)

1. **Position is address; relatedness is geometry + relations.** Compare on the *right face* (face-aware), not
   the whole vector.
2. **Numbers relate via the homomorphism, NEVER via relation edges.** Operand↔result / number↔number edges
   pollute and erase prior lessons. Arithmetic = faces; it writes no relation edges. The **operation is
   CLASSIFIED by the GRU op head from context** (`PredictQuery`), not a symbol→op regex — so `x` means
   multiply only when context (flanking operands) says so, and the homomorphism then computes.
3. **Identity faces are frozen; free faces learn.** A migration (e.g. `"one"`→value 1) happens in the free
   numeric face; the char identity stays. Migration is real but unstable under repulsion and the semantic
   face is lexical, so learned associations are carried by relations, not geometric retrieval.
4. **Equivalence ≠ format.** Value-equivalence (2≡two) vs surface-conversion ("1"→"one" must produce the word)
   — grade accordingly (see `AnswerEquivalence`).
5. **The substrate is uniform: objects, relations, compositions, functions are all elements.** Prefer adding a
   *kind of element* over a parallel data structure.

---

## 7. Relations and shapes as positioned elements

A relation is a positioned element, not a side-graph entry:
- `RelationElementNode` is a first-class **positioned element** — position = centroid of its endpoints' faces
  (derived, projected by `GetRelationElements`), strength = `1 − SynthesisContradiction`. Being positioned, a
  relation can itself be a relation endpoint (higher-order).
- `_relationIndex` is the O(1) **access index** over relation-elements — the same role `_nodes` plays for
  concept-elements — not a graph on top. The mutable payload (contradiction dynamics + lifecycle counts) is
  load-bearing for `ObserveContradiction`/`ReinforceEvidence`, snapshots, and pruning; keyed mutation needs an
  index, which every spatial store has.
- Retrieval (the Relational tier + `TryRelationElementNeighbour`) is the **canonical element-native traversal**,
  reading each element's `Strength`.

Composer shapes follow the same pattern (§3): `ElementKind.Function` elements in `_functionElements`, with
`RelatedTo` to the shape-elements they compose, executed by traversal. Objects, relations, compositions, and
functions are all positioned elements indexed for access — one uniform substrate.
