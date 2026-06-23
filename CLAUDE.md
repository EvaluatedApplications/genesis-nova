# Genesis-Nova — working agreement

Genesis-Nova builds **one conscious field** — a structured platonic substrate the neural layer reasons *with* (not
an LLM that stores knowledge in weights). Orientation, in order: **`PLATONIC_MIND.md`** (the founding vision — read
first), `PLATONIC_THEORY.md` (formal substrate model), `PLATONIC_NUCLEUS.md` (the dual-face data model),
`PLATONIC_CONSCIOUSNESS.md` (the self / vital loop, as built), `PLATONIC_PRIOR_ART.md` (disclosure); plus
`README.md` (spec), `PROJECT_GLIDER.md` (composer), `claude/README.md` (the agent CLI tools + memory).

## The substrate & how it reasons

The platonic space is a geometric substrate the neural layer reasons WITH (see `PLATONIC_MIND.md`): identity in the
frozen faces, meaning as a distributional cloud in the large face, reasoning as relaxation of the field. Read
`PLATONIC_NUCLEUS.md` (the dual-face data model) before touching the faces.
- **Faces** (`Core/FaceLayout.cs`): each concept is a vector split into regions — **poly `[0,21)`** (a number's
  value, add/sub-homomorphic, `e[i]=v·10^-(i+1)`), **log `[21,42)`** (same value, mul/div-homomorphic,
  `ln|v|·10^-(i+1)`), **char `[42,202)`** (spelling, generative), **word `[202,dim)`** (whole tokens,
  recognition). Identity dims are frozen (numbers are ground truth); free dims learn.
- **Exact arithmetic rides the homomorphism**: `poly(a)+poly(b)=poly(a+b)`, `log(a)+log(b)=log(a·b)` —
  computed in the geometry, exact, generalizes to unseen operands, no stored facts (log face has no 0).
- **Relations are positioned elements** (centroid of endpoints, strength `1−contradiction`); learned
  associations (`one↔1`, `apple→fruit`) are retrieved by relation traversal. **Numbers never form relation
  edges** — that pollutes and erases prior lessons (hard rule).
- **Reasoning = inference routing** (`Infer/GenesisInferenceEngine.cs`): the GRU route head picks a path and
  each path *abstains/falls through* if it can't answer — GRU-query arithmetic (op classified from context →
  homomorphism) → composer plan (plan head picks a composition shape → a glider runs it on the substrate) →
  learned-function (a transform vector selected by relation) → relation-first retrieval → concept-chain →
  neural fallback. Every answer carries a `DecisionPath`; inspect it with `claude/GenesisInspect`.
- Capability **emerges from composing substrate ops**; the GRU only chooses *which* — never a hardcoded
  answer table or symbol parser.

## Golden paths for training

Proven recipes — follow them; they encode hard-won lessons (see the `nova-*` memories):
- **Test-first, demonstrate-can-emerge.** Write the emergence test first; assert a majority bar AND that
  answers route via the platonic path (not the neural fallback). Show capability *can* emerge — never overfit
  a targeted test or hardcode an answer.
- **Mastery-gated regime, not fixed step counts.** Train a lesson to mastery (held a stability window), anneal
  LR near target, and **rehearse mastered lessons** to resist forgetting. Don't burn fixed-count loops.
- **Bare > diverse surfaces; always hold out** operands/entities — consistent filler becomes a spurious
  correlate; held-out instances prove compute/retrieve, not memorization.
- **Supervision is DERIVED from each example's structure, never hardcoded** (`ResolveQueryLabel` /
  `ResolveRouteLabel` / `ResolvePlanLabel`): op classified from context, shape from the output's structure.
- **Numbers compute via the homomorphism, never relation edges.** Equivalence ≠ format — grade by VALUE
  (`AnswerEquivalence`), bidirectionally (`2≡two`).
- **Add a kind of element, not a parallel data structure.** Push compute into the space (faces / relations /
  compositions / functions); keep the GRU thin (a few logits per shape).
- **Heavy training/emergence tests are `[SlowFact]` (opt-in `RUN_SLOW=1`);** the default suite is fast
  behaviour tests. After training, use `GenesisInspect report|probe` to see what the model actually does.
- **Known failure mode:** generalized/autonomous training erodes the arithmetic ROUTER (mis-routes `a+b` to
  retrieval) while the homomorphism stays intact — a *routing* problem, not compute. Catch it with
  `GenesisInspect probe`; fix with arithmetic-route supervision, not more flat epochs. See
  `[[nova-generalized-training-routing]]`.

## Memory: save proactively — don't wait to be asked

The file memory (`MEMORY.md` + its files) is the durable record. Compaction and new sessions drop exact
detail, so the moment something durable is established, **write it** as you work:
- a non-obvious conclusion from an audit/investigation (e.g. "REPL inference == training: one shared engine"),
- a gotcha and its fix (e.g. "spawn console with `2>nul` literal, not `2^>nul` — the caret disables the redirect"),
- a user preference, decision, or correction,
- where something lives (file → symbol/API) that cost a search to find.

The test, before you re-read a file or re-derive something: *"would future-me have to look this up again after
compaction?"* If yes, it belongs in memory. One fact per file; update the matching file rather than
duplicating; delete memories proven wrong. This is the cheapest token win available — re-reading large files
to recover dropped detail is the dominant avoidable cost.

## Memory: write it the way Genesis-Nova is strong

> Historical note (2026): a `claude/ClaudeMemory` daemon once kept a live GRU-routed associative index over
> `MEMORY.md` (auto-started each session, `recall`/`enqueue`/`watch`/`metrics`). **That tool was RETIRED** —
> continuous training is now hosted in the GenesisNova desktop app's **gym** (see `[[nova-gym]]`), and
> `claude/GenesisInspect` is the only remaining CLI (read-only diagnostics: `report`/`query`/`probe`/`geometry`/
> `gymprobe`). The `recall`/`enqueue`/index commands no longer exist; `MEMORY.md` is read directly.

The file memory IS the source of truth — write it well:

- **Keyword-rich one-line descriptions** in `MEMORY.md` — include the terms (and synonyms) you would later search
  by, not just a title.
- **Link relationally** with `[[other-memory]]` — liberal links let a reader reach neighbouring memories.
- **Associations/structure over blobs.** Keep content in the file; keep the description associative; keep
  `MEMORY.md` correct and **delete memories proven wrong**.
- **Never encode number↔number facts as relations.** The substrate's hard rule — numeric relation edges pollute
  and erase prior lessons; arithmetic is the homomorphism, not stored edges.
