# Platonic Shapes — substrate-executed compositions the composer selects

> Principle: **the GRU is a thin selector; the work happens in the platonic space.** An ability adds
> *platonic* compute, not *GRU* compute. A "shape" is a composition the substrate executes by its own rules
> (R2 compose, the numeric homomorphism, relation hops, concatenate-compose); the GRU only chooses which
> shape — an O(1) plan-head class. This extends what is already offloaded — homomorphic arithmetic,
> relations-as-elements, learned-function transforms — up into the COMPOSITION layer.

## The composer

A GRU **plan-head** (`GenesisNeuralModel.PredictPlan`) classifies the composition shape; the inference route
`TryGenerateFromGliderPlan` assembles the corresponding glider and runs it on the substrate. Supervision is
derived from each example's structure (`GenesisTrainer.ResolvePlanLabel`), never hardcoded. The plan head
grows by a few logits per shape — the only GRU cost. It abstains (returns false) when the plan head is
untrained or a shape is unavailable, so the other inference routes and their decision-path contracts are
unaffected.

## Where compute lives

- **GRU:** an N-way shape classifier. Tiny — a few logits per shape, nothing else.
- **Platonic space:** all of it — homomorphic arithmetic, N-way fold, relational retrieval,
  concatenate-binding, recursive shape-of-shapes.

---

## The shapes

### Fold (variadic reduce) — a Composition element
`a+b+c+…` / `a·b·c·…` for any operand count. The substrate performs an N-operand **R2 compose**
(`PlatonicGliderInterpreter.ComposeArithmetic` → `TickExecutor.ExecuteTick(Compose)`): poly-sum = +,
log-sum = ×, decoded by the homomorphism. Plan kinds `fold-sum` / `fold-product`; the GRU picks one, the
substrate reduces N operand-elements. The reduce *is* the homomorphism over N elements. Supervision: the
output equals the sum (→ fold-sum) or product (→ fold-product) of ≥3 operands.

### Const / scale (×k, +k) — a learned-function transform
"double" / "triple" / "+k" is a learned transform vector `T(f)` applied by composition (`embed(x)+T(f)`),
selected by relation — the **learned-operation route** (`TransformAccumulator`). Nothing computes in the
GRU; the substrate applies the transform. The `Const` block also supplies fixed scalars inside other shapes
(e.g. the ×2 in the Ref shape below).

### Seq (multi-step / scaffold) — a Concatenate-Composition element
A response is a `CompositionMode.Concatenate` of part-elements: a scaffold text-chunk bound to a
substrate-computed value. The scaffold is **mined from graded-correct outputs** into a **chunk-element
store** (`PlatonicSpaceMemory.MineChunk` / `TryGetTopChunk`, keyed by `SeqScaffoldTag`) — training targets
are correct by construction, so a Seq-structured output (scaffold words + the operands' sum) contributes its
scaffold; `GenesisTrainer.MineComposerShapes` does the mining. At inference the Seq shape **retrieves the
most-reinforced scaffold** and binds it to a `Fold(Add)` value via the interpreter's `Seq` block (=
`CompositionMode.Concatenate` semantics). The scaffold is learned, not a template baked into inference; the
shape abstains until one is mined. This is the cache/binding idea done element-natively.

### Expression-chain (multi-operator) — chaining compute-elements by context, NO locked cue

The general "complex chaining of elements". A MULTI-operator expression (`2 x 7 + 3` → `17`) is solved by
chaining compute-elements: the plan head selects the expression-chain shape (kind 8); the route
(`TryGenerateFromExpressionChain`) parses the operand/operator sequence and classifies **each operator from
CONTEXT** via the learned op head on its local binary window — there is **no symbol→op map**, so `x` is
multiply only because of the operands around it (in `let x = 5` the same token is a variable). It evaluates
with standard precedence (× ÷ before + −, each pass left-to-right); **every binary step is one substrate R2
compose + homomorphic decode**, so the answer generalises to any operands. Control flow lives in the route;
the compute is on the substrate. Supervision is derived from the expression's own value
(`GenesisLabelResolver.IsExpressionChain`, an oracle); no invented cue token is ever required — this replaced
the old `twicelarger` Ref cue (a single token locked to one prebuilt glider — see `nova-no-token-locking`).

### Ref (higher-order, glider-of-gliders) — a Function element referencing shape-elements [RETAINED INFRA]

> The Ref block + `PlatonicShapeRegistry` + `ElementKind.Function` registry remain as general substrate
> machinery (a glider can still invoke a named glider recursively, tested by `PlatonicShapeMachineryTests`),
> but the plan head no longer SELECTS a named ref shape via a cue word — kind 8 is now the expression-chain.
Shapes are first-class **`ElementKind.Function`** elements of the space
(`PlatonicSpaceMemory.RegisterFunctionElement` / `FunctionElements`): positioned (composed embedding) with a
`RelatedTo` pointing at the shape-elements they compose, held in their own index (`_functionElements`),
parallel to `_nodes` (Object) and `_relationIndex` (Relation), so they never contaminate concept retrieval.
`PlatonicShapeRegistry` is the catalogue — it holds each shape's glider definition (a composition of the
primitive block vocabulary, not a premade answer) and materializes the matching Function element. `larger` =
max(a,b) (Compare→Branch); `twicelarger` = 2·max = `Compute(Multiply, Ref("larger"), Const(2))`, a Function
element whose `RelatedTo` is `larger`. The interpreter carries the registry library, so a `Ref` block
executes by traversing + composing the referenced shape recursively (`EvalRef`). The GRU selects the top
shape; the substrate recurses and computes exactly for any operands. The substrate is uniform — objects,
relations, compositions, functions — all elements, all substrate-executed.

### Deferred arithmetic / retrieval
Digit-arithmetic (homomorphism) and retrieval (relations) compute in the substrate already; they are reached
by dedicated routes (`GruQuery` / relation-first) rather than the composer, to keep their existing
decision-path contracts. Unifying them under the plan-head would move no compute into the GRU either way.

---

## The architectural shape of it
Glider blocks are C# records run by a deterministic interpreter (control flow in the interpreter, compute on
the substrate). Shapes are platonic elements; "running a glider" is the substrate composing / executing
elements via its physics (`ElementKind.Function`, `CompositionMode.Concatenate`, `TickExecutor` compose).
The GRU never grows beyond shape-selection.

## Plan kinds and supervision

| Kind | Shape | Substrate mechanism | Structural supervision |
|---|---|---|---|
| 1 | arithmetic (digit) | homomorphism (via `GruQuery`) | ≥2 numeric operands, numeric output |
| 2 | predicate | Compare→Branch (difference sign) | output ∈ {greater, less, equal} |
| 3 | retrieval | relation hop (via relation-first) | non-numeric operands → non-numeric output |
| 4 | arithmetic → word | Hop(Compute, Word) | ≥2 operands, number-word output |
| 5 | fold-sum | N-way R2 compose (poly-sum) | sum of ≥3 operands |
| 6 | fold-product | N-way R2 compose (log-sum) | product of ≥3 operands |
| 7 | seq | mined scaffold ∘ Fold(Add) | scaffold words + operands' sum |
| 8 | expression-chain | per-operator op-head classification + precedence eval, chaining R2 composes | ≥2-operator expression whose precedence value == output |

Tests via main-code `Generate` (assert correctness *and* the `glider-plan` decision path): `PredicateComposerTests`,
`FormatComposerTests`, `FoldComposerTests`, `SeqComposerTests`, `RefComposerTests`.
