# Self-improvement loop — the genesis engine for Claude

> Direction (`PLATONIC_RECKONING.md`): iterations should target **SUBTRACTION toward the keep-core** (the
> homomorphism, the distributional face, composition-by-reuse, relaxation, abstention) — *not* adding another
> head, plan shape, or skill.

Use Genesis-Nova + the file memory (`MEMORY.md`, read directly) + the inspector (`GenesisInspect`) as a
**compounding loop** that improves four things, each measurable:

1. **Prompt** — `CLAUDE.md` and `claude/*.md` (how cold-start-me behaves).
2. **Platonic structure** — substrate capabilities (faces, relations, compositions, functions; the relaxation route).
3. **Training** — regimes, creators, structure-derived supervision.
4. **Inference** — the substrate settles by its own confidence; abstains when nothing does.

Each iteration is small, measured, and recorded to file memory so gains compound and regressions are caught.

## The cycle (run every iteration)

1. **Recall + measure.** Read `MEMORY.md` directly to surface relevant memories (+ their `[[ ]]` see-also
   links); read them. `GenesisInspect report|probe` (or `gymprobe <level>` for the gym's per-skill curriculum
   reading) to measure current capability. Ground the iteration in evidence, not assumption.
2. **Target.** Pick ONE weakness on ONE axis — prefer the highest-leverage *measured* failure.
3. **Hypothesize.** One concrete change + the metric that will show whether it worked.
4. **Change.** Implement minimally; build green; no broad speculative rewrites (one consolidated build).
5. **Measure.** Re-run the metric and compare before→after **honestly**. A null/negative result is a result —
   record it, don't bury it.
6. **Record.** Write the outcome (what changed, the numbers, worked/didn't + why) to file memory — one fact
   per file, linked with `[[ ]]`, and update `MEMORY.md`. Distill durable direction into `CLAUDE.md`/docs.
7. Repeat.

## How to measure each axis

| Axis | Metric | Tool |
|---|---|---|
| Prompt | does cold-start-me recall the right memory + follow the golden paths | read `MEMORY.md`, read `CLAUDE.md` |
| Platonic structure | relaxation settles substrate-exact (homomorphism + distributional face); abstains when nothing does | `DialecticalSpace.Reason` / `platonic-reason` route + emergence test |
| Training | held-out accuracy + retention of prior lessons | emergence test (`RUN_SLOW`) |
| Inference | the substrate settles by its own confidence (`DecisionPath`), abstains otherwise | `GenesisInspect query|probe` |

## Discipline (the hard-won rules — see the `nova-*` memories)

- Demonstrate-can-emerge; never hardcode or overfit a targeted test.
- Numbers never form relation edges; arithmetic is the homomorphism. Equivalence ≠ format.
- Heavy training/emergence tests are opt-in (`RUN_SLOW`); don't burn GPU on fixed-count loops.
- File memory (`MEMORY.md`) is the source of truth, read directly. Record direction so it compounds.

## Where iteration outcomes live

Not in this file — in **file memory** (`MEMORY.md` + its files). Each iteration appends/updates a `nova-*`
memory and links it with `[[ ]]`. This doc is the *process*; the memories are the *log*. Start an iteration by
reading the most relevant `nova-*` memory and the current `GenesisInspect probe`.
