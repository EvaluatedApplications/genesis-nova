# Project Glider

> The platonic space is Conway's Game of Life. We are building gliders.

## 0. The metaphor, made precise

The platonic space is a **substrate with fixed physics**:

- **Concepts** with geometric faces (numeric poly/log, char, word, free region).
- **Relations** — learned equivalence/association edges (`one ↔ 1`, `apple ↔ fruit`).
- **The numeric face homomorphism** — exact symbolic arithmetic, no per-fact storage.
- **Storage/retrieval** — a concept (even a multi-token chunk) is a first-class addressable object.

A **glider** is a small, stable, **reusable** configuration in that substrate that does useful
composite work *by referencing those objects* — never by memorising input→output products. A glider
for "what is X plus Y → the answer is Z" costs *one scaffold + the equivalence edges + the
homomorphism* and answers **every** X, Y. That is `O(structure)`, not `O(facts)`.

**The program:**

1. **Hand-build** the most efficient glider for a target task and **demonstrate** it executes on the
   substrate physics (no NN). — *Done for the answer-template; see §2.*
2. Make that hand-built glider the **training target**: teach the GRU to **construct** the glider
   itself from the input, deriving supervision from each example's own structure.
3. **Generalise** to higher-order objects: gliders that select/compose other gliders (glider guns).

This is the concrete unit of the north-star mission — *train the platonic INTERFACE, not an LLM; the
NN learns to compress, retrieve, and compute in the platonic space.*

## 1. The abilities a glider exercises

Target: `"what is one plus one"` → `"the answer is 2"`. Four decisions, each a reusable primitive:

| Ability | What it is | Substrate mechanism | Status |
|---|---|---|---|
| **SCAFFOLD** | retrieve the stored chunk `"the answer is ⟨slot⟩"` | concept storage / relation | atoms exist; chunk concept new |
| **RESOLVE** | operand → canonical numeric (`"one"→1`) | relational neighbor lookup | proven (`number-word-equiv`) |
| **COMPUTE** | `1+1=2` | numeric face homomorphism | proven exact (`PlatonicArithmetic_IsExact`) |
| **FORMAT** | render slot as `2` or `two` (a conditional) | equivalence edge, reversed | proven |

"Produce chunks of text / conditionals / concats" = SCAFFOLD / FORMAT-choice / scaffold⊕slot.

## 2. What exists now (the hand-built glider)

- **`src/GenesisNova/Cognition/PlatonicGlider.cs`**
  - `PlatonicGlider(Name, Steps)` — the structure.
  - `GliderStep` = `EmitConcept(conceptName)` | `EmitComputed(GliderOp, operandSlots, GliderFormat)`.
  - `GliderOp {Add,Subtract,Multiply,Divide}`, `GliderFormat {Digit,Word}`.
  - `PlatonicGliderInterpreter` — deterministic "physics" that runs a glider: RESOLVE via
    `GetNeighbors(Relational)`, COMPUTE via `GetFreshNumericEmbedding` + decode, FORMAT via the
    reversed equivalence edge.
- **`Tests/PlatonicGliderDemoTests.cs`** — pure-platonic, production face dim, fast. One glider yields
  `the answer is 2` (`one+one`), generalises to the never-seeded `9+9 → the answer is 18`, mixes
  surfaces, switches to word format (`the answer is two`), and composes subtraction — all on the
  substrate, hand-composed.

This proves the substrate hosts the structure efficiently. **Now we teach the GRU to build it.**

## 3. The training goal — GRU constructs gliders

Extend the GRU's `PredictQuery` from `(op, operandMask)` to a fixed-shape **PLAN tuple**:

```
plan = { scaffoldId,  op + operandMask,  format ∈ {digit, word, passthrough} }
```

Two small extra heads on the existing query head. By making the scaffold carry the fixed text and
exactly one computed slot, we get composition **without** a variable-length program decoder (hard to
train, easy to break). The genuinely new learning is the three heads *coordinating*.

### Supervision is DERIVED, never hardcoded (the discipline of `ResolveQueryLabel`)

Every training pair auto-yields its own glider label from its structure:

- **scaffold label** — the output's constant skeleton with the numeric answer factored out
  (`"the answer is ⟨n⟩"`); the variable position is the slot. Derive by aligning a creator's outputs.
- **op / operands** — from the input's numeric structure (already done).
- **format** — whether the target's slot token is a digit or a number-word (we have the vocab).

So `("what is one plus one" → "the answer is 2")` yields `{scaffold:"the answer is ⟨n⟩", op:add,
operands:[t,t], format:digit}` with no hand-authoring. The GRU learns `text → glider`; the
interpreter executes it; we **demonstrate** the capability can emerge, we do not overfit.

## 4. Training registers (the roadmap)

Each register is an `IExampleCreator` that varies everything *except* the structure, forcing the GRU
to generalise the glider. Build and prove them in order; each is a test-first emergence demo at
production face dimension with held-out instances.

| # | Register | Teaches | Generalises over | Status |
|---|---|---|---|---|
| **R1** | `corenova:answer-template` | the answer-template glider | operands (digit/word), surface framings, answer format, op (add→div) | **built (§5)** |
| R2 | `corenova:answer-scaffold-choice` | scaffold is a *retrievable variable*, not a fixed prefix | multiple scaffolds (`the answer is`, `that gives`, `= ⟨n⟩`, bare `⟨n⟩`) → GRU must *select* | planned |
| R3 | `corenova:format-decision` | the FORMAT conditional, cued by the prompt | "in words" / "as a number" cues drive digit-vs-word | planned |
| R4 | `corenova:retrieval-answer` | the SAME glider over *non-arithmetic* slots | `"what is the capital of france" → "the answer is Paris"` — scaffold + retrieval, no compute | planned |
| R5 | `corenova:predicate-answer` | conditional/branching gliders | `"is 5 greater than 3" → "yes"` — compute a predicate, branch the chunk | planned |
| R6 | `corenova:multi-step` | **higher-order** gliders (glider guns) | `"what is one plus one plus one"`, two-stage compute-then-convert → gliders that call gliders | planned |

Design rules for every register (carried from the proven bootstrap findings):

- **Bare-ish, varied surfaces.** Consistent filler becomes a spurious correlate; vary it or drop it.
- **Held-out instances** in the demo (operands/entities never trained) to prove *compute/retrieve*,
  not memorise.
- **Mastery = via the platonic plan** (used-platonic, not neural fallback), held a stability window.
- **Deterministic** per `(creator, difficulty)`; difficulty widens scope, not just count.

## 5. R1 — the first register (built)

**`src/GenesisNova/Data/Creators/GliderAnswerCreator.cs`** — `corenova:answer-template`
(complexity 28, PromptAnswer). Emits framed-arithmetic → `"the answer is Z"` with:

- operands rendered as **digits or number-words** (RESOLVE must handle both),
- multiple **input framings** per op (`what is X plus Y`, `X + Y`, `the sum of X and Y`, …),
- answer rendered as **digit or word** (FORMAT conditional),
- difficulty widening the **operator set** (d0 add → d3 add/sub/mul/div) and range.

`corenova:` marks the *kind* (tool-training); complexity 28 orders it **after** its sub-skills
(`number-word-equiv`, computation) in the focused curriculum. Unit-tested in `CoreBootstrapTests.cs`
(`GliderAnswerCreatorTests`). **DEFERRED from `ExampleCreatorRegistry.All`** (2026-06-13) until the GRU
plan-heads exist — without them it can't construct the glider via the platonic path, so it would stall
an autonomous run; re-add to `All` once §3 is built.

> NOTE: until the GRU plan-heads (§3) exist, this register trains via the *existing* paths (arithmetic
> short-circuit + decode), which can already partially produce `"the answer is Z"`. That is a useful
> baseline; the glider-construction capability arrives with the plan-heads + emergence demo.

## 6. Curriculum, interference, and the long run

- The focused autonomous planner trains by complexity, one creator to depth, replaying mastered ones
  (forgetting fix + drive-to-depth already landed).
- **Known risk — shared-parameter interference.** Later training erodes earlier heads (~18% measured:
  arithmetic dragged `number-word` retention 100%→82%). It is *not* relational-graph pollution
  (suppressing operand↔result edges left retention unchanged) — it is shared GRU/output-head/embedding
  drift. The planner's auto-re-open mitigates it but risks primitive↔composite **ping-pong**. Multi-head
  plan training (§3) will feel this most; watch for it, and consider replay-weighting / lower late-LR.
- The platonic faces and relations carry the load that the NN otherwise would; the more the glider
  offloads to the substrate, the less the NN weights have to hold, and the less interference bites.

### 6.1 Per-component block regimens — components-first (2026-06-14)

Rather than train the full answer-template glider (which needs the §3 plan-heads), we train each
reusable **block** on its own, so gliders later compose from already-trained components. Each regimen is
`{ focused creator + inference route + can-emerge test }`, mirroring how `Hop` (`number-word-equiv`,
`retrieval-category`) and `Compute` (`arithmetic:*`) were already proven.

The mechanism — `PlatonicGliderInterpreter.TryResolveCapability(input)`: parse a **compact** form and
answer it by running a small **hand-built block composition** on the substrate. The blocks ARE the
inference mechanism; there is no GRU-constructed plan. Used by **both** the inference engine
(`TryGenerateFromGliderBlock`, in the route-1/2 dispatch → credited platonic) and the trainer
(`ResolveRouteLabel` labels these **route-1/direct** whenever the block path reproduces the target —
exact and space-independent from step 0, so the router gets clean supervision).

| Block | Creator | Compact form → answer | Hand-built glider (composition) |
|---|---|---|---|
| **Compare** | `numeric:compare` (22) | `compare a b` → `greater\|less\|equal` | `Branch(Compare(>) , "greater", Branch(Compare(<), "less", "equal"))` |
| **Branch** | `numeric:larger` (23) | `larger a b` → the larger operand | `Branch(Compare(>, op0, op1), op0, op1)` |
| **Const** | `numeric:scale` (24) | `double\|triple x` → `x*2 \| x*3` | `Compute(Multiply, [op0, Const(k)])` |
| **Ref** | `numeric:twice-larger` (26) | `twicelarger a b` → `2*max(a,b)` | `Compute(Multiply, [Ref("larger"), Const(2)])` — higher-order |

All four are in `ExampleCreatorRegistry.All` (active). `Ref` is the **higher-order** component (a glider
that invokes another glider — a "glider gun"): its top glider references the named `larger` sub-glider
from the interpreter's default library (`BuildDefaultLibrary`) and doubles the result, so the regimen
exercises composition-by-reference, not inlining. `Operand`/`Literal` ride along inside these (operand
reads, the predicate/scaffold labels). **Limitation:** the `scale` and `twice-larger` regimens keep the
multiplied value `>= 1` — the multiplicative (log) face has no representation for zero (`log 0`
undefined); Compare/larger use the additive face and handle 0 fine.

**Seq and Literal are intentionally NOT given a regimen** (measured reason — must be beneficial): `Seq`
is the space→text output boundary, already exercised by every multi-token creator; `Literal` (emit a
stored chunk) is covered by the `Hop` retrieval primitive. A standalone creator for either would be
contrived.

**Block list status: COMPLETE.** Every block now has a real training path (Hop, Compute, Compare,
Branch, Const, Ref) or a reasoned skip (Seq, Literal; Operand rides along). The one remaining glider
item is a *substrate-purity* refinement, not a missing component: making whole gliders first-class
`PlatonicElement`s so `Ref` is element-native (the R6 "whole-glider-as-element"), the analog of the
Compute/Compare element-native refactors. The §3 GRU plan-heads (the model CONSTRUCTING arbitrary
gliders = the full answer-template) remain deferred by design — components-first, not full-glider.

**Tests:** `Tests/GliderBlockRegimenTests.cs` — (1) `BlockCapabilities_ResolveExactly_OnTheSubstrate`
(fast, no NN: each composition resolves exactly), (2) `BlockCapabilities_CanEmerge_ViaTraining` (trained
on the focused creators, the model routes each capability to its platonic block path and answers
correctly — majority, via the block path).

## 7. Emergence tests (how we know it worked)

For each register, a production-dimension demo asserting the GRU **constructs the right glider**
(right scaffold + computed/retrieved slot + format) for held-out instances **via the platonic plan**,
majority bar (demonstrate-can-emerge, not certainty). Held-out operands/entities prove generalisation.
The hand-built `PlatonicGliderInterpreter` is the oracle: the GRU's constructed plan should match (or
the executed output should match) the hand-built glider's output.

## 8. Open questions

- Can three plan-heads **coordinate** reliably, or do they need staged training (op → +format →
  +scaffold)?
- Does scaffold **selection** (R2) form a clean relational carrier the way bare retrieval did (16/16)?
- Does the SAME glider transfer to **retrieval** slots (R4) with no arithmetic — proving it isn't
  arithmetic-specific?
- How far do **higher-order** gliders (R6) compose before the fixed-shape plan tuple is insufficient
  and we genuinely need a (bounded) sequence decoder?

## 9. Block vocabulary & platonic-form status (2026-06-13)

The glider block set (`src/GenesisNova/Cognition/PlatonicGlider.cs`), and which blocks are **element-native
platonic forms** vs **measured keeps**. Principle: only the blocks that genuinely *compute* something
become substrate elements; orchestration/boundary blocks stay in the interpreter (the universal "physics"
rule). Each refactor keeps the meta-layer behaviour as an oracle and is verified identical.

| Block | Role | Status |
|---|---|---|
| `Compute(op,args)` | arithmetic | **element-native** → R2 `Composition` element via `TickExecutor`; subtract/divide compose with the **complement** (`¬b`). Oracle: `ComputeDirect`. |
| `Compare(op,l,r)` | predicate | **element-native** → sign of the difference `Composition` (reuses Compute's complement). **Regimen:** `numeric:compare` (§6.1). |
| `Hop(src,target)` | relational retrieval | **keep** — already substrate: it *follows a relation edge*; that edge IS the platonic form. **Regimen:** `number-word-equiv`, `retrieval-category`. |
| `Branch(cond,a,b)` | conditional select | **keep** — control flow (the evaluator = the universal rule). Its substrate content is the `Compare` predicate, element-native. **Regimen:** `numeric:larger` (§6.1). |
| `Seq(parts)` | text assembly | **keep** — the space→text output boundary, not a platonic composition. No regimen by design (§6.1). |
| `Operand`,`Const`,`Literal` | leaves | **keep** — already substrate concepts / inputs. `Const` regimen: `numeric:scale` (§6.1); `Literal` covered by `Hop`. |
| `Ref(name)` | higher-order invoke | **regimen done** (`numeric:twice-larger`, §6.1) — a glider invokes a named sub-glider from the default library (glider gun). Remaining *refinement* (not a missing component): the element-native *whole-glider-as-element* (R6 hof — a `Composition`/`Function` element whose parts are other glider-elements; needs gliders to be `PlatonicElement`s), analogous to the Compute/Compare element-native refactors. |

**Face-aware distance (the real fix behind "use the unused dims").** The numeric/char/word faces are the
**generative encoding coordinate system** — decode any coordinate → a number/char/text chunk, *including
unseen* — so they stay clean; **relatedness lives in the relation graph + a face-scoped distance**, never
whole-vector proximity. Implemented (genesis `EuclideanDistanceRange` parity): the semantic KNN (`VPTree`)
compares only the semantic face `[WordFaceStart..dim)`; `GetNearestConcepts` uses the genesis **hybrid**
(`min(semantic, arithmetic)` for numeric queries, semantic-only for text). Freezing non-function dims at
zero was a *bandage* for un-face-scoped operations; face-scoping is the fix. Remaining whole-vector ops
(R2 `SumEmbeddings`, the message-pass nudge's identity-restore) are not distance, so out of scope here.

**Verified:** light suite green; heavy demos — digit→word 10/10 platonic, GRU framed 5/5, regime
all-converged with retention 100/100/85/84% (improved). Per-component block regimens (§6.1) added &
verified for Compare/Branch/Const/Ref (substrate-exact + can-emerge via the block path). The block layer
is COMPLETE — every block has a regimen or a reasoned skip. Remaining glider items are both OPTIONAL /
deferred-by-design: the element-native whole-glider-as-element refinement (R6), and the §3 GRU
plan-heads (full-glider construction, deferred — components-first).
