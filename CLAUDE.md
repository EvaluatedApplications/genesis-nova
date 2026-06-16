# Genesis-Nova — working agreement

Genesis-Nova trains a small GRU to **use** a structured platonic substrate (compress / retrieve / compute),
not to be an LLM. Orientation: `README.md` (spec), `PLATONIC_SPACE.md` (substrate), `PROJECT_GLIDER.md` +
`PLATONIC_SHAPES.md` (composer), `claude/README.md` (the agent CLI tools + memory),
`claude/SELF_IMPROVE.md` (the compounding loop you run to iterate on prompt / structure / training / inference).

## The substrate & how it reasons

The platonic space is a geometric substrate; the GRU is a **thin selector**, not the reasoner — the work
happens in the space. Read `PLATONIC_SPACE.md` before touching it.
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

## Memory: fill it the way Genesis-Nova is strong

A **continuous-mastery** Genesis-Nova daemon (`claude/ClaudeMemory`) keeps an **associative index over
`MEMORY.md`** — it rehearses the current set toward a held-out accuracy bar and logs a learning curve, then
idles. It auto-starts **visibly each session** (a logon Startup shortcut + the `SessionStart` hook), runs on
**GPU + licensed** (EvalApp adaptive tuning — see `[[nova-evalapp-license]]`), and redirects libtorch's stderr
to `.claude-nova/daemon.stderr.log` so its window shows only the cycle curve. The file memory is the source of
truth (exact); the index is a fuzzy **GRU-routed** pointer — use it to find *which memory files to open*.
Observe it live with `ClaudeMemory.exe watch` (status + curve) or `metrics`.

**How to use it (run these — don't just read about them):**
```powershell
# At the START of a non-trivial task — GRU-routed retrieval (Generate): the model routes the query + returns
# the relevant memory key. Then open memory/<key>.md. The index points; the files hold the truth.
.\claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe recall "<topic words>" 2>$null

# Push an ad-hoc association without editing files (instant; the daemon trains it in the background):
.\claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe enqueue "<cue>" "<memory-key>" 2>$null

# Observe the daemon's learning (live monitor, or the raw curve rows):
.\claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe watch      # status + curve, refreshing
.\claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe metrics    # held-out acc / acc / route / conf

# After you SAVE a memory or edit MEMORY.md, the auto-started daemon rebuilds itself — nothing to run.
# If the tool isn't built yet: dotnet build claude/ClaudeMemory -c Release  (it then auto-starts next session).
```

Write memories the way the substrate is strong:

- **Keyword-rich one-line descriptions.** The index derives its query keywords from each entry's
  `description`, so recall coverage == the words you put there. Include the terms (and synonyms) you would
  later search by, not just a title.
- **Link relationally** with `[[other-memory]]`. Relations are Nova's native structure; liberal links let
  recall reach neighbouring memories.
- **Associations/structure over blobs.** Keep content in the file; keep the description associative.
  Generate-don't-store: the index is *recomputed* from `MEMORY.md` (the daemon rebuilds on change), so stale
  facts drop automatically — keep `MEMORY.md` correct and let the index regenerate; never hand-curate the index.
- **Never encode number↔number facts as relations.** The substrate's hard rule — numeric relation edges
  pollute and erase prior lessons; arithmetic is the homomorphism, not stored edges.

The index is **lossy and forgets** — depend on it only to *point*; the files hold the truth. Commands and the
auto-start hook are documented in `claude/README.md`.
