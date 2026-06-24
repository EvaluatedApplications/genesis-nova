# Project Glider

> The platonic space is Conway's Game of Life. Gliders are the reusable structures built on it.

> ⚠️ **Read with `PLATONIC_RECKONING.md` (2026-06).** The reckoning splits this doc in two. The *general
> substrate primitives* (§0–§1: Compute/Fold/Compare via the homomorphism, Hop as a relation edge, composition
> by reuse) are KEPT — they generalize. But the **GRU "plan head" that classifies one of 9 pre-built composition
> SHAPES, its `ResolvePlanLabel` supervision, and the per-shape glider routes (§2, §3–§6) are flagged as
> OVERFITTING to the gym taxonomy** — a task-classifier over a fixed menu, the thing to SUBTRACT, not grow. Treat
> §2 onward as a record of the machinery being shed, not the current direction. (`PLATONIC_SHAPES.md`, referenced
> below, has been removed.)

The principle: **if the GRU can figure something out, it should build it in the platonic space** — from
faces, relations-as-elements, R2 composition, and learned-function transforms. Capability EMERGES from
composition; it is never enumerated. Hand-built compositions that lock tokens to fixed meanings, and
templated input→output answers, are not part of this — the block set is a *composable vocabulary*, and the
GRU learns to select and sequence it.

## 0. The metaphor, made precise

The platonic space is a **substrate with fixed physics**:

- **Concepts** with geometric faces (numeric poly/log, char, word, free region).
- **Relations** — learned equivalence/association edges (`one ↔ 1`, `apple ↔ fruit`), positioned elements.
- **The numeric face homomorphism** — exact symbolic arithmetic, no per-fact storage.
- **Storage/retrieval** — a concept (even a multi-token chunk) is a first-class addressable element.

A **glider** is a small, stable, **reusable** configuration that does composite work *by referencing those
elements* — never by memorising input→output products. A glider for "what is X plus Y → the answer is Z"
costs *one scaffold chunk + the equivalence edges + the homomorphism* and answers **every** X, Y. That is
`O(structure)`, not `O(facts)`.

The unit of the north-star mission: *train the platonic INTERFACE, not an LLM; the NN learns to compress,
retrieve, and compute in the platonic space.*

## 1. The block vocabulary (`Cognition/PlatonicGlider.cs`)

A glider is a single root `GliderBlock`; blocks nest (a block's inputs are other blocks). The interpreter
(`PlatonicGliderInterpreter`) is the deterministic substrate physics — no NN — and evaluates a glider against
the query's operand tokens.

| Block | Role | Substrate mechanism |
|---|---|---|
| `Operand(i)` | read the i-th operand token | input |
| `Literal(chunk)` | emit a stored text chunk (scaffold) | concept storage |
| `Hop(src, target)` | follow the strongest relation to a related concept of a kind | relation edge (`GetNeighbors`) |
| `Compute(op, args)` | fixed-arity arithmetic | numeric face homomorphism (R2 compose) |
| `Fold(op, from)` | reduce op across all operands from a slot | one N-way R2 compose |
| `Compare(op, l, r)` | predicate (yields a boolean) | sign of the difference composition |
| `Branch(cond, a, b)` | conditional select | control flow over a `Compare` |
| `Seq(parts)` | concatenate child outputs into text | `CompositionMode.Concatenate` |
| `Const(k)` | a literal numeric constant | input |
| `Ref(name)` | invoke another named glider on the same operands | recursive execution over the shape registry |

`Compute`/`Compare`/`Fold` are **element-native**: arithmetic is a first-class `Composition` element built by
the substrate's R2 rule (`TickExecutor`, `ExecuteTick(Compose)`) and decoded via the homomorphism;
subtract/divide compose with the **complement** (`¬b = −embed(b)`), so `a−b = a + ¬b` (poly) and `a/b = a·¬b`
(log). The meta-layer direct sum is kept as an oracle and verified identical. `Hop` is already substrate —
it follows a relation edge, which IS the platonic form. `Branch`/`Seq` are control-flow/boundary blocks (the
evaluator is the universal "physics" rule); their substrate content is the `Compare` predicate and the
concatenate binding.

## 2. The composer — the GRU constructs gliders

The GRU's **plan head** (`GenesisNeuralModel.PredictPlan`) classifies the composition SHAPE for an input; the
inference route `TryGenerateFromGliderPlan` assembles the corresponding glider and runs it on the substrate.
The blocks are the vocabulary, the GRU is the composer, and nothing is hardcoded per token. Supervision is
**derived from each example's own structure** (`GenesisTrainer.ResolvePlanLabel`) — e.g. an output of
`greater`/`less`/`equal` is a predicate shape; an output equal to the sum of ≥3 operands is a fold-sum; a
scaffold-words-then-number output is a Seq. The plan head grows by a few logits per shape and abstains when
untrained, so the other inference routes keep their decision-path contracts.

The shapes and their substrate mechanisms (the `PLATONIC_SHAPES.md` catalogue has been removed; see the
reckoning note above on why this shape-set is being subtracted). In brief: predicate
(Compare→Branch), arithmetic→word (Hop∘Compute), fold-sum/product (N-way compose), Seq (mined scaffold ∘
Fold), Ref (a `Function` element referencing other shape-elements). Digit-arithmetic and retrieval compute on
the substrate too but are reached by their dedicated routes (`GruQuery` / relation-first).

## 3. Shapes as elements

Two refinements make the substrate uniform — objects, relations, compositions, and functions are all
positioned elements:

- **Seq's scaffold is mined, not baked in.** Graded-correct training outputs feed a **chunk-element store**
  (`PlatonicSpaceMemory.MineChunk` / `TryGetTopChunk`); inference retrieves the most-reinforced scaffold and
  binds it to a substrate-computed value. The chunk is learned from positive results, not a template string.
- **Ref's shapes are `ElementKind.Function` elements.** `PlatonicShapeRegistry` registers named shapes
  (`larger` = max via Compare→Branch; `twicelarger` = `Multiply(Ref("larger"), Const(2))`) both as executable
  glider definitions and as positioned Function elements whose `RelatedTo` points at the sub-shapes they
  compose. The interpreter resolves a `Ref` recursively against the registry library — a composition of
  compositions executed by the substrate.

## 4. Emergence tests

Each shape has a production-dimension test that trains on the focused structure and asserts the model
**constructs the right shape** for held-out instances **via the plan path** (`glider-plan` decision path), a
majority bar (demonstrate-can-emerge, not certainty). Held-out operands prove the substrate computes rather
than memorises: `PredicateComposerTests`, `FormatComposerTests`, `FoldComposerTests`, `SeqComposerTests`,
`RefComposerTests`. `PlatonicGliderDemoTests` verifies each block on the real substrate, no NN.

## 5. Curriculum and interference

- The focused autonomous planner trains by complexity, one creator to depth, replaying mastered ones (the
  forgetting fix and drive-to-depth).
- **Shared-parameter interference** is the live risk: later training can erode earlier heads (shared
  GRU/output-head/embedding drift — not relational-graph pollution; suppressing operand↔result edges leaves
  retention unchanged). The planner's auto-re-open mitigates it but risks primitive↔composite ping-pong;
  multi-head plan training feels it most. The faces and relations carry the load the NN otherwise would — the
  more a glider offloads to the substrate, the less the weights hold and the less interference bites.

## 6. Open questions

- How reliably do the plan and query heads **coordinate** as the shape set grows?
- Does scaffold **selection** form a clean relational carrier when multiple scaffolds compete?
- Does the same composer machinery transfer to richer **retrieval** slots (non-arithmetic), proving it isn't
  arithmetic-specific?
- How far do **higher-order** shapes (Ref of Ref) compose before a fixed plan tuple needs a bounded sequence
  decoder?
