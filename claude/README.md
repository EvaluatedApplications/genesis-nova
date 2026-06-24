# Claude tools for Genesis-Nova

The CLI an agent (Claude) uses to **inspect** the trained Genesis-Nova model. `GenesisInspect` is a standalone
console app that references the `GenesisNova` project (like `bench/RaceBench`); it is **not** part of
`GenesisNova.slnx`, so it never affects the test suite.

```
claude/
  GenesisInspect/   diagnostic CLI — what the trained model IS + how it answers (the only CLI)
```
Runtime data (not source) lives at the repo root in **`.claude-nova/`**: the model checkpoint
(`genesis-nova.autosave.checkpoint.json`) and the `interaction-log.jsonl`.

**Memory:** the flat file memory (`MEMORY.md` + its files) is the durable source of truth and is **read
directly** — there is no daemon or associative index. Continuous skill training lives in the GenesisNova
desktop app's **gym** (see `[[nova-gym]]`), not a background CLI.

## GenesisInspect

Strictly **read-only** inspection of a trained checkpoint. By DEFAULT it inspects the model in
`<repo>/.claude-nova/` (the NN + its `.platonic.json` companion); pass `--state-dir` to inspect a
different model — e.g. the gym/autonomous trainer at `%LocalAppData%/GenesisNova/models/`.

```bash
dotnet run --project claude/GenesisInspect -c Release -- report                    # architecture + substrate + capacity
dotnet run --project claude/GenesisInspect -c Release -- query "find diagnostic cli"  # output, decision path, activation
dotnet run --project claude/GenesisInspect -c Release -- probe                     # capability battery (✓/✗ where known)
dotnet run --project claude/GenesisInspect -c Release -- space three               # deeper activation around a concept
# flags: --cpu, --state-dir <dir> (which model; default .claude-nova), --checkpoint <file>
```

## How Claude uses this (convention)

- Read `MEMORY.md` directly at the start of a task as a cheap "what do I already know near this?" — the file
  memory is the source of truth; open the linked memory file(s) it points to.
- After learning durable facts: write them to the file memory (`MEMORY.md` + a file), linked with `[[ ]]`.
- `GenesisInspect report`/`probe` to check the model's current capabilities before trusting a route.

## Notes for others

- Requires the same TorchSharp + CUDA setup as the main project (CPU fallback works; GenesisInspect defaults
  to CPU). Build it once with `dotnet build claude/GenesisInspect -c Release` then run the `.exe` directly.
- `.claude-nova/` is regenerable runtime state; treat it as a cache, not source (ignore it in VCS).
- libtorch prints benign `.grad` warnings to stderr during training; suppress with `2>$null` (PowerShell).
