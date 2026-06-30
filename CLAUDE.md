# Genesis-Nova: working agreement

Genesis-Nova builds **one conscious field**, a structured platonic substrate the neural layer reasons *with* (not
an LLM that stores knowledge in weights). Orientation: **`PLATONIC_MIND.md`** (the founding vision; hold its
narrative lightly), `PLATONIC_THEORY.md` (formal substrate model), `PLATONIC_NUCLEUS.md` (the dual-face data model),
`PLATONIC_CONSCIOUSNESS.md` (the self / vital loop, where mechanisms are real and "consciousness" aspirational),
`PLATONIC_PRIOR_ART.md` (disclosure); plus `README.md` (spec), `claude/README.md` (the agent CLI tools + memory).

**The direction:** the substrate's generalizing core is real (the numeric homomorphism, the distributional word
face, composition-by-reuse, relaxation, learned transforms), so build on it. A task-classifier over a fixed gym
taxonomy (a PLAN head with per-shape outputs, structural label resolvers, the route head used as a classifier, the
neural decoder as a primary answer path) is the thing to SUBTRACT, not extend. Don't add the next head/shape/skill.
The work is toward a single substrate-driven control path (answer by the substrate's own confidence; abstain when
nothing settles) trained by self-supervised prediction.

## The substrate & how it reasons

The platonic space is a geometric substrate the neural layer reasons WITH (see `PLATONIC_MIND.md`): identity in the
frozen faces, meaning as a distributional cloud in the large face, reasoning as relaxation of the field. Read
`PLATONIC_NUCLEUS.md` (the dual-face data model) before touching the faces.
- **Faces** (`Core/FaceLayout.cs`): each concept is a vector split into regions. **Poly `[0,21)`** (a number's
  value, add/sub-homomorphic, `e[i]=v·10^-(i+1)`), **log `[21,42)`** (same value, mul/div-homomorphic,
  `ln|v|·10^-(i+1)`). At the **production face dim (≥512, default 1024)** the **address-space layout** takes over
  (`FaceLayout.IsAddressSpace`): **kind `[42,48)`**, **spelling `[48,208)`**, **structure `[208,400)`**, **op
  `[400,416)`** — the frozen, invertible identity **codec `[0,416)`** — plus a learned **orbital** tail `[416,dim)`
  (the meaning cloud). (Below dim 512 the legacy layout applies: char `[42,202)` / word `[202,dim)`.) Identity dims are
  frozen (numbers are ground truth); the orbital/free dims learn. NB **two distinct widths**: this is the **substrate
  face** width (`FaceDimension`, 1024 in production); the **GRU controller** width (`HiddenSize`) is separate — 512
  default / 2048 in the app — and the two are decoupled (the model has no face-dim-sized params).
- **Exact arithmetic rides the homomorphism**: `poly(a)+poly(b)=poly(a+b)`, `log(a)+log(b)=log(a·b)`,
  computed in the geometry, exact, generalizes to unseen operands, no stored facts (log face has no 0).
- **Relations are positioned elements** (centroid of endpoints, strength `1−contradiction`); learned
  associations (`one↔1`, `apple→fruit`) are retrieved by relation traversal. **Numbers never form relation
  edges**; that pollutes and erases prior lessons (hard rule).
- **Retrieval is O(log N) via the lattice.** `DialecticalSpace` indexes concepts in a VP-tree
  (`Cognition/PlatonicLattice.cs`) over the semantic face; `GetNearestConcepts`/`Reason` harvest a bounded
  candidate set from it and **re-score live faces** (hybrid: exact scan below 384 nodes, lattice above, the same
  result, just faster). `ElementStore.ActiveCount` is O(1) because it gates this hot path.
- **Reasoning = inference routing** (`Infer/GenesisInferenceEngine.cs`): the route ladder, where each path
  *abstains/falls through* if it can't answer. GRU-query arithmetic (op classified from context → homomorphism)
  → composer plan → learned-function → **relaxation (`platonic-reason`)** → geometric/relation retrieval →
  concept-chain → neural fallback. Every answer carries a `DecisionPath`; inspect it with `claude/GenesisInspect`.
- **Keep-core control path** (`GenesisNovaConfig.KeepCoreControl`, the app turns it on; default-off = byte-identical
  legacy path). When on: training labels + route/plan/op perception anchor on the **discriminative cue** (shared
  `Cognition/PlatonicConceptAnchors`, matching what inference retrieves on, so trainer and inference must extract the
  anchor the same way); **relaxation (`DialecticalSpace.Reason`) is the primary retrieval route**; and a
  non-arithmetic query that nothing settles **ABSTAINS** (`platonic-abstain`) instead of emitting a neural
  hallucination. The plan/route/op heads + label resolvers are overfit machinery to subtract, so don't extend them.
- Capability **emerges from composing substrate ops**; the GRU only chooses *which*, never a hardcoded
  answer table or symbol parser.

## Golden paths for training

Proven recipes; follow them, as they encode hard-won lessons (see the `nova-*` memories):
- **Test-first, demonstrate-can-emerge.** Write the emergence test first; assert a majority bar AND that
  answers route via the platonic path (not the neural fallback). Show capability *can* emerge. Never overfit
  a targeted test or hardcode an answer.
- **Mastery-gated regime, not fixed step counts.** Train a lesson to mastery (held a stability window), anneal
  LR near target, and **rehearse mastered lessons** to resist forgetting. Don't burn fixed-count loops.
- **Bare > diverse surfaces; always hold out** operands/entities. Consistent filler becomes a spurious
  correlate; held-out instances prove compute/retrieve, not memorization.
- **Supervision is DERIVED from each example's structure, never hardcoded** (`ResolveQueryLabel` /
  `ResolveRouteLabel` / `ResolvePlanLabel`): op classified from context, shape from the output's structure.
  ⚠️ `ResolvePlanLabel` + the route-head-as-classifier are **overfit to the gym taxonomy**, reverse-engineering
  task type from output structure to supervise a classifier. The direction is **self-supervised prediction**, not
  more structural label resolvers. Under `KeepCoreControl` the route label + perception anchor on the
  **discriminative cue** (`PlatonicConceptAnchors`, shared with inference) so the controller sees the real
  geometry (see `[[nova-keepcore-landed]]`). Do not add another label resolver / plan kind / output shape.
- **Numbers compute via the homomorphism, never relation edges.** Equivalence ≠ format, so grade by VALUE
  (`AnswerEquivalence`), bidirectionally (`2≡two`).
- **Add a kind of element, not a parallel data structure.** Push compute into the space (faces / relations /
  compositions / functions); keep the GRU thin (a few logits per shape).
- **Heavy training/emergence tests are `[SlowFact]` (opt-in `RUN_SLOW=1`);** the default suite is fast
  behaviour tests. After training, use `GenesisInspect report|probe` to see what the model actually does.
- **Known failure mode:** generalized/autonomous training erodes the arithmetic ROUTER (mis-routes `a+b` to
  retrieval) while the homomorphism stays intact. That is a *routing* problem, not compute. Catch it with
  `GenesisInspect probe`; fix with arithmetic-route supervision, not more flat epochs. See
  `[[nova-generalized-training-routing]]`.

## Memory: save proactively, don't wait to be asked

The file memory (`MEMORY.md` + its files) is the durable record. Compaction and new sessions drop exact
detail, so the moment something durable is established, **write it** as you work:
- a non-obvious conclusion from an audit/investigation (e.g. "REPL inference == training: one shared engine"),
- a gotcha and its fix (e.g. "spawn console with `2>nul` literal, not `2^>nul`, since the caret disables the redirect"),
- a user preference, decision, or correction,
- where something lives (file → symbol/API) that cost a search to find.

The test, before you re-read a file or re-derive something: *"would future-me have to look this up again after
compaction?"* If yes, it belongs in memory. One fact per file; update the matching file rather than
duplicating; delete memories proven wrong. This is the cheapest token win available: re-reading large files
to recover dropped detail is the dominant avoidable cost.

## Memory: write it the way Genesis-Nova is strong

> There is no live memory-indexing daemon. Continuous training is hosted in the GenesisNova desktop app's **gym**
> (see `[[nova-gym]]`), and `claude/GenesisInspect` is the only CLI (read-only diagnostics: `report`/`query`/
> `probe`/`geometry`/`gymprobe`). There are no `recall`/`enqueue`/index commands; `MEMORY.md` is read directly.

The file memory IS the source of truth, so write it well:

- **Keyword-rich one-line descriptions** in `MEMORY.md`. Include the terms (and synonyms) you would later search
  by, not just a title.
- **Link relationally** with `[[other-memory]]`; liberal links let a reader reach neighbouring memories.
- **Associations/structure over blobs.** Keep content in the file; keep the description associative; keep
  `MEMORY.md` correct and **delete memories proven wrong**.
- **Never encode number↔number facts as relations.** This is the substrate's hard rule: numeric relation edges pollute
  and erase prior lessons; arithmetic is the homomorphism, not stored edges.
