# Claude tools for Genesis-Nova

CLI tools an agent (Claude) uses to **inspect** the trained Genesis-Nova model and to keep an **associative
index** over its own file memory. Both are standalone console apps that reference the `GenesisNova` project
(like `bench/RaceBench`); they are **not** part of `GenesisNova.slnx`, so they never affect the test suite.

```
claude/
  GenesisInspect/   diagnostic CLI — what the trained model IS + how it answers
  ClaudeMemory/     associative index over the file memory (Nova as a pointer, not the store)
  truth/            optional repo-local fact source (fallback when no MEMORY.md index is given)
```
Runtime data (not source) lives at the repo root in **`.claude-nova/`**: the memory checkpoint
(`genesis-nova.autosave.checkpoint.json`) and the `interaction-log.jsonl`.

## The memory model (read this first)

The **flat file memory is the source of truth** — exact, durable, never-forgetting. Genesis-Nova is **not**
the store; it is a *fuzzy associative index* that, for a vague query, points back to the relevant memory
**key(s)**, which Claude then opens. Content stays in files; Nova only provides the "you know something near
X → look in *these* memories" reach that a keyword grep misses.

Why not store content in Nova? It is lossy and forgets under continual training (measured — see
`GenesisInspect`). So we use its strength (associative reach) and avoid its weakness (faithful recall).

**Truth-table principle:** the index is a *pure function* of `MEMORY.md`. `rebuild` reads the index, derives
`keyword → memory-name` associations from each entry's description, and trains a **fresh** memory — so a
renamed/deleted memory drops its stale association automatically on the next rebuild. Generate, don't store.

This doubles as the project's most honest **retention benchmark**: the metric is "does `recall` still return
the right keys as the index grows?" The day it stops is the data point the substrate work needs.

## GenesisInspect

Strictly **read-only** inspection of a trained checkpoint. By DEFAULT it inspects the ClaudeMemory daemon's
model in `<repo>/.claude-nova/` (the NN + its `.platonic.json` companion); pass `--state-dir` to inspect a
different model — e.g. the autonomous trainer at `%LocalAppData%/GenesisNova/models/`.

```bash
dotnet run --project claude/GenesisInspect -c Release -- report                    # architecture + substrate + capacity
dotnet run --project claude/GenesisInspect -c Release -- query "find diagnostic cli"  # output, decision path, activation
dotnet run --project claude/GenesisInspect -c Release -- probe                     # capability battery (✓/✗ where known)
dotnet run --project claude/GenesisInspect -c Release -- space three               # deeper activation around a concept
# flags: --cpu, --state-dir <dir> (which model; default .claude-nova), --checkpoint <file>
```

## ClaudeMemory

Associative index over the file memory, on a **separate** Nova instance (`.claude-nova/`) so it never touches
the autonomous curriculum model.

**Async / decoupled (preferred):** run the daemon once and it trains in the **background** — watching the
index file and a fact queue — so you never block on training.
```bash
# Run the background trainer (in its own window); Ctrl+C to stop & save. Logs each action it takes.
dotnet run --project claude/ClaudeMemory -c Release -- serve --index "<path-to-MEMORY.md>"

# Drop a fact instantly (no model load, no waiting — the daemon trains it next tick):
dotnet run --project claude/ClaudeMemory -c Release -- enqueue "widget alpha" "claude-widget-key" [relate]
# ...or just edit MEMORY.md — the daemon notices and rebuilds fresh.
```
**One-shot / read (no daemon needed — file-backed, works cold):**
```bash
dotnet run --project claude/ClaudeMemory -c Release -- rebuild --index "<path-to-MEMORY.md>"  # recompute fresh
dotnet run --project claude/ClaudeMemory -c Release -- recall  "router confidence"            # → memory key(s)
dotnet run --project claude/ClaudeMemory -c Release -- stats | log 20
# Or set CLAUDE_MEMORY_FILE once instead of passing --index every time.
# flags: --index <file>, --interval N (serve), --gpu (default CPU), --reps N, --dir <state-dir>
```
The daemon recomputes FRESH when the index changes (so stale associations drop) and trains queued facts
incrementally. State is file-backed (`.claude-nova/`), so `recall`/`rebuild` also work cold without it.

### Auto-start each session

`.claude/settings.json` registers a **SessionStart hook** that runs `claude/start-memory-daemon.ps1`, which
starts the daemon in the background **once per session, idempotently** (if a `serve` process is already
running it does nothing — no duplicates). On startup the daemon *resumes* the saved checkpoint unless
`MEMORY.md` changed since the last save (then it rebuilds), so repeated session starts are cheap.

- Index resolution: `$CLAUDE_MEMORY_FILE` → the known per-user `MEMORY.md` → repo `claude/truth/memory.truth`.
  Set `$CLAUDE_MEMORY_FILE` to point it elsewhere.
- Requires the tool built once: `dotnet build claude/ClaudeMemory -c Release` (the hook no-ops with a note if
  the exe is missing — it never fails session start).
- Stop it with `Get-Process ClaudeMemory | Stop-Process`. It runs hidden; actions are in
  `.claude-nova/interaction-log.jsonl` (`ClaudeMemory log`).

**Index source resolution:** `--index <file>` → `$CLAUDE_MEMORY_FILE` → `claude/truth/memory.truth`
(repo-local fallback so others can use the tool without an external memory index). The index parser accepts
both the `MEMORY.md` line shape (`- [name](file.md) — description`) and a plain `cue => response` /
`a <=> b` truth file. Tool locations under `claude/` are always re-derived from the live filesystem.

## How Claude uses these (convention)

- Keep `serve` running in the background (one window) so training is async — never block a task on it.
- `recall` at the start of a task as a cheap "what do I already know near this?" — treat the result as a
  **pointer** to a file to open, not as the answer.
- After learning durable facts: write them to the file memory (`MEMORY.md` + a file); the daemon rebuilds,
  or `enqueue` for an instant drop. (Cold, no daemon: `rebuild`.)
- `GenesisInspect report`/`probe` to check the model's current capabilities before trusting a route.

## Notes for others

- Requires the same TorchSharp + CUDA setup as the main project (CPU fallback works; the memory tool defaults
  to CPU). Build a tool once with `dotnet build claude/<Tool> -c Release` then run the `.exe` directly.
- `.claude-nova/` is regenerable runtime state; treat it as a cache, not source (ignore it in VCS).
- libtorch prints benign `.grad` warnings to stderr during training; suppress with `2>$null` (PowerShell).
