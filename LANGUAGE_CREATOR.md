# Language Creator — spec

A framework for building **small, trainable languages** on the Genesis-Nova substrate. Instead of writing an
interpreter, you *declare* a language — a set of operations ("tool calls"), each with **one canonical syntax** —
and **train** the GRU to route them and the substrate to execute them. Nothing is hardcoded: routing and
execution emerge from examples; adding an operation means adding examples, never code.

This generalizes the `claude/ClaudeMemory` experiment (which is one such language with a single `recall`
operation) into a reusable capability. It is the practical form of the project thesis: *Genesis-Nova is a new
kind of code-writer that likes very specific queries — it should have syntax.* (See `README.md`,
`PLATONIC_SPACE.md`, `PROJECT_GLIDER.md`.)

> **Scope & isolation.** This is **self-contained** and has **nothing to do with the research bootstrap /
> autonomous curriculum** (`CoreBootstrapSuite`, `GenesisAutonomousTraining*`, the platonic-interface research
> mission). A language trains **only** on its own declared operations, in its **own checkpoint** (its
> `LocalStateDirectory`, e.g. `.claude-nova/` for the Claude tools) — it never loads, runs, or depends on the
> curriculum model. The bootstrap/autonomous training is a separate concern and is **not run** for tooling
> built on this spec. The Claude integration (`claude/ClaudeMemory`, `claude/GenesisInspect`) is one
> **claude-tools-only** instance of this framework; it shares the substrate *code* but never the curriculum
> *training* or its checkpoint.

---

## 1. Why this is not an LLM prompt layer

An LLM maps fuzzy natural language → intent and tolerates a thousand phrasings. The substrate is the opposite:
it is a **programming language**. It wants **one exact syntax** and generalizes the **operand**, exactly as
arithmetic trains one form `a + b` and generalizes to unseen operands — you never teach six ways to phrase
"add". The win is *operand generalization*, not phrasing memorization.

Consequences that shape this whole spec:

- **One canonical surface form per operation.** Not paraphrases. Diversity goes in the *operands*, which are
  held out to prove generalization (the "bare > diverse surfaces; always hold out" golden path).
- **The operation is selected by routing, executed by the substrate.** The GRU's job is to pick *which*
  operation and *which* route; the work (compute / retrieve / associate / compose) happens in the platonic
  space. The GRU stays thin.

---

## 2. The one invariant that makes or breaks it

> **The op-token is a ROUTE TRIGGER, never a relation participant.**

The verb that names an operation (`recall`, `sum`, `define`) exists to tell the GRU route head *which operation
this is*. It must **not** form relation edges to operands or results.

This is the same rule as **"numbers never form relation edges"** (`PLATONIC_SPACE.md`): a token that recurs in
*every* example of an operation, if allowed to edge to results, accumulates a strong edge to whatever target is
most frequent and **overwhelms the operand** — every query collapses to one answer. We observed exactly this:
adding `recall {topic}` as `find <topic> → key` relation facts made `find` edge to every memory and pulled
`forgetting`, `router erosion`, and `benchmark` all to the single largest memory. The op-token polluted
retrieval precisely as number↔number edges pollute arithmetic.

In arithmetic this is already done right: `+` routes to the homomorphism; it never edges to the numbers. Every
user operation must obey the same separation:

| concern        | mechanism                                  | forms relation edges? |
| -------------- | ------------------------------------------ | --------------------- |
| op-token       | GRU **route head** (which operation)       | **never**             |
| operands       | faces (compute) / relations (associate)    | only if `kind` says so |
| result         | produced by the route, not stored on a verb| never via the verb    |

---

## 3. Anatomy of an operation

A *language* is a set of **operations**. Each operation declares only:

- **`name`** — the operation id and its route label.
- **`syntax`** — ONE canonical surface form with typed operand slots, e.g. `recall {topic}`,
  `sum {a:num} {b:num}`, `define {term} as {meaning}`.
- **`kind`** — the semantics class, which selects the substrate route used to execute it:

  | kind        | substrate route (`Infer/GenesisInferenceEngine.cs`)        | generalizes via            |
  | ----------- | ---------------------------------------------------------- | -------------------------- |
  | `compute`   | GRU-query → face homomorphism (poly/log)                   | operand value (exact)      |
  | `retrieve`  | relation-first / concept-chain                             | operand→target association |
  | `associate` | relation formation between operands (an edge)              | structural                 |
  | `compose`   | composer plan-head → glider runs a shape on the substrate  | sub-step composition       |
  | `transform` | learned-function vector selected by relation, applied by composition | few-shot function    |

- **`examples`** — operand-varied instances. Supervision is **DERIVED** from these and from `kind`
  (`ResolveRouteLabel` / `ResolveQueryLabel` / `ResolvePlanLabel`), never hand-written as an answer table.
- **`holdout`** — operand instances trained on *never*, used to measure generalization.

The parser splits `syntax` into `op-token + operand slots`. That split is **presentation only** — it does not
choose an answer. The GRU routes; the substrate executes.

---

## 4. File format

A language is one declarative file, `*.lang`. Editing it *is* the API — no code changes.

```lang
language memory

op recall
  syntax:  recall {topic}
  kind:    retrieve
  examples:
    recall forgetting -> nova-retention-diagnosis
    recall benchmark  -> nova-benchmark
    recall mission    -> nova-north-star
  holdout:
    recall router-erosion -> nova-generalized-training-routing

op sum
  syntax:  sum {a:num} {b:num}
  kind:    compute            # rides the poly-face homomorphism; exact on unseen operands
  examples:
    sum 3 4   -> 7
    sum 12 30 -> 42
  holdout:
    sum 87 56 -> 143

op define
  syntax:  define {term} as {meaning}
  kind:    associate          # forms a term<->meaning relation edge; the verb does NOT edge to anything
  examples:
    define apple as fruit
    define spark as ignite
```

Rules the loader enforces:

- exactly one `syntax` per op (reject multiple surface forms — that's the LLM mistake);
- the op-token is registered route-only (it is excluded from relation-edge formation);
- `compute` operands must be `:num` typed (they ride faces, never edges) — numeric relation edges are rejected;
- the training set is a **pure function** of the `.lang` file (+ a live queue), so deleting an op drops its
  associations on the next rebuild.

---

## 5. Training protocol

Per the golden paths (`README.md` §"Golden paths for training"):

1. **Derive supervision from structure.** Route label from `kind`; query/plan labels from the operand/output
   structure; never a hardcoded map.
2. **One syntax, varied operands, held out.** Train each op on its operand-varied examples; keep `holdout`
   operands unseen. Generalization = held-out operand accuracy, not training accuracy.
3. **Mastery-gated, not fixed step counts.** Train an op until a stability window holds; anneal LR near target.
4. **Rehearse to resist forgetting.** Interleave mastered ops while learning new ones — relations and routes
   interfere, so rehearsal is mandatory (this is why a single pass produces mushy recall).
5. **Numbers compute, never edge.** `compute` ops use the homomorphism; grade by VALUE (`AnswerEquivalence`,
   `2≡two`), bidirectionally.

---

## 6. The continuous daemon

The current `serve` daemon is **change-driven**: one fixed-reps pass when the definition changes, then it idles.
That under-trains (we have never trained this for long). The language-creator daemon is **continuous**:

- **Always training the most up-to-date set.** It rebuilds the training set from the current `.lang` file (+
  queue), then keeps training/rehearsing it — when there is nothing new, it does **not** sleep; it **rehearses
  toward mastery** and re-draws operand instances. Capability therefore *improves the longer it runs*.
- **Definition change → fold in.** New/edited ops are added, removed ops are dropped (pure function of the
  file). Mastered ops keep rehearsing so they don't regress.
- **Queue → incremental operands.** `enqueue` adds operand instances live (no model load), trained next tick.
- **Checkpoint + metrics each cycle.** Persist the model and append a metrics line so improvement is
  **observable** (§7).

Mastery loop (sketch):

```
load language(.lang) + queue           -> ops, examples, holdouts
while running:
    fold in changes (add/drop/edit ops)
    for op in ops (ordered: weakest first, with rehearsal of mastered):
        train an operand-varied batch
        score op on its HELD-OUT operands  -> accuracy, route-purity, confidence
    persist checkpoint
    append metrics(cycle, per-op held-out accuracy/route-purity/confidence)
    # no sleep while any op is below its mastery bar
```

---

## 7. Observability — "we should observe improvement"

Because the daemon trains the current set continuously, the per-op **held-out** metrics should climb over
cycles. The daemon emits a metrics log (`.../metrics.jsonl`) with, per cycle and per op:

- **held-out accuracy** — fraction of `holdout` operands answered correctly (the headline generalization
  number);
- **route purity** — fraction routed via the intended platonic route (not the neural fallback);
- **confidence** — mean platonic confidence on held-out operands.

A `lang curve <op>` command prints the accuracy-over-cycles series so the learning curve is visible. The success
criterion for this whole feature is a **rising held-out curve** on operands the model was never trained on —
operand generalization under one fixed syntax.

---

## 8. Non-goals / guardrails

- **No phrasing variety.** One syntax per op. If you want robustness to wording, that is the caller's job
  (normalize to the syntax), not the model's.
- **No hardcoded dispatch.** The runtime never pattern-matches `syntax` to select an answer. `syntax` is only a
  parser hint; the GRU routes and the substrate executes. Adding capability = adding examples.
- **No op-token relation edges.** Enforced by the loader (§2). Violating it reintroduces the collapse bug.
- **No number↔number edges** for `compute` ops (substrate hard rule).
- **The GRU is not optional.** It is the generalizer for unseen operands — hand-wiring routes/answers is
  "writing in assembly", i.e. hardcoding. The GRU is the compiler from intent to substrate composition.

---

## 9. Mapping onto existing code

| spec element            | existing home                                                            |
| ----------------------- | ------------------------------------------------------------------------ |
| operation routing       | GRU route head + `ResolveRouteLabel` (`src/.../Train`)                   |
| execution per `kind`    | `Infer/GenesisInferenceEngine.cs` routes (homomorphism / relation-first / concept-chain / glider plan / learned-function) |
| derived supervision     | `GenesisLabelResolver` (`ResolveQueryLabel` / `ResolvePlanLabel`)        |
| compose `kind`          | composer plan-head + gliders (`PROJECT_GLIDER.md`, `PLATONIC_SHAPES.md`) |
| transform `kind`        | learned-function route (`TransformAccumulator` / `FoldPathDiscovery`)    |
| continuous daemon       | generalize `claude/ClaudeMemory` `serve` (change-driven → mastery loop)  |
| metrics / inspection    | `claude/GenesisInspect` (`report` / `probe`)                             |

---

## 10. Phased build

- **Phase 0 — fix the primitive.** Make the `recall` op-token route-only (stop it forming key edges); confirm
  recall stops collapsing to one key. (Direct consequence of §2.)
- **Phase 1 — `.lang` loader.** Parse `language`/`op`/`syntax`/`kind`/`examples`/`holdout`; build the training
  set as a pure function of the file; enforce loader rules (§4).
- **Phase 2 — continuous mastery daemon + metrics.** Replace the change-driven loop with the §6 mastery loop;
  emit `metrics.jsonl`; add `lang curve`.
- **Phase 3 — wire all `kind`s.** `compute` / `associate` / `compose` / `transform` end-to-end with derived
  supervision.
- **Phase 4 — generalization eval.** Held-out operand harness per op; the rising curve (§7) is the acceptance
  test. Capture in a `[SlowFact]` emergence test.

Success: a user writes a `.lang` file, the daemon trains it continuously, and the held-out operand curve climbs
— a new operation taught by example, routed and executed by the trained substrate, with no code written.
