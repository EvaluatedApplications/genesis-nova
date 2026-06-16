# Genesis-Nova — setup guide

**Genesis-Nova is the source of truth for everything in this repo.** The engine in `src/GenesisNova` is the
*general* substrate (a small GRU controller over a structured platonic space — see `README.md` for the spec and
`CLAUDE.md` for the working agreement). Everything else builds on it and must not fork it:

- `src/GenesisNova/` — **the engine** (GENERAL; no app-specific hardcoding). The single source of truth.
- `Tests/` — the behaviour + emergence suite (in `GenesisNova.slnx`).
- `bench/RaceBench/` — equal-param benchmark vs a transformer (references the engine; not in the .slnx).
- `claude/` — **applied** tooling on top of the engine (NOT in the .slnx, so it never affects the test suite):
  - `GenesisInspect/` — read-only diagnostic CLI: what a trained model *is* and how it answers.
  - `ClaudeMemory/` — a continuous-mastery Genesis-Nova **index** over a file memory (`MEMORY.md`).
  - `truth/` — repo-local fact source (fallback when no external `MEMORY.md` is given).
- `.claude-nova/` — **runtime state only** (the daemon's checkpoint + logs, ~400 MB). Generated, gitignored,
  regenerable. Treat as a cache, never as source.

## Prerequisites

- **.NET 8 SDK** (Windows; the projects target `net8.0-windows`).
- **TorchSharp** pulls libtorch automatically on restore. **CPU works everywhere**; an NVIDIA **GPU + CUDA** is
  optional and used by the daemon (`--gpu`) and benchmarks. libtorch prints benign `.grad` warnings to stderr
  during training — suppress with `2>$null` in PowerShell.
- Git (any recent build; a Visual Studio install bundles one under `Common7/.../Git/cmd`).

## 1. Build the engine + run the fast tests

```powershell
dotnet build GenesisNova.slnx -c Release
dotnet test  GenesisNova.slnx                 # FAST behaviour suite — the default regression gate
```

Heavy training/emergence tests are opt-in `[SlowFact]` — run them only when needed and targeted:

```powershell
$env:RUN_SLOW = "1"; dotnet test --filter "FullyQualifiedName~GruQueryConstruction"; $env:RUN_SLOW = $null
```

> Never run the full slow suite as a regression — it is long and not the gate. The fast `dotnet test` is.

## 2. Build the applied tools

They reference the engine but are **not** in `GenesisNova.slnx`, so build each directly once, then run the exe:

```powershell
dotnet build claude/ClaudeMemory   -c Release
dotnet build claude/GenesisInspect -c Release
```

## 3. Set up the memory daemon

The ClaudeMemory daemon keeps a fuzzy, GRU-routed **associative index over `MEMORY.md`**. The **file memory is
the source of truth** (exact, durable); the index only points you at *which memory file to open*. It trains
continuously in the background, rebuilds itself when `MEMORY.md` changes, and idles when mastered.

**Auto-start (recommended):** `.claude/settings.json` registers a Claude Code **SessionStart hook** that runs
`claude/start-memory-daemon.ps1`, which launches the daemon once per session, idempotently (no duplicate if one
is already serving). It resumes the saved checkpoint unless `MEMORY.md` changed (then it rebuilds).

- The hook path in `.claude/settings.json` is **absolute** (`C:\Users\<you>\genesis-nova\claude\start-memory-daemon.ps1`)
  — adjust it for your checkout.
- **Index resolution:** `$CLAUDE_MEMORY_FILE` → the known per-user `MEMORY.md` → repo `claude/truth/memory.truth`.
  Set `$CLAUDE_MEMORY_FILE` to point it at your own memory file.

**Run / observe / query manually:**

```powershell
$mem = "$env:USERPROFILE\.claude\projects\C--Users-dongy\memory\MEMORY.md"   # your MEMORY.md (or $env:CLAUDE_MEMORY_FILE)
$exe = "claude\ClaudeMemory\bin\Release\net8.0-windows\ClaudeMemory.exe"

& $exe serve --gpu --index $mem 2>$null     # background trainer (own window); Ctrl+C stops & saves
& $exe recall "router confidence" 2>$null   # GRU-routed recall → memory key(s); then open memory/<key>.md
& $exe enqueue "widget alpha" "widget-key"   # drop an ad-hoc association instantly (trained next tick)
& $exe watch                                 # live status + learning curve;  `metrics` / `log 20` also work
```

`recall`/`rebuild` are file-backed and work **cold** (no daemon needed). See `claude/README.md` for the full
command reference and the daemon's training regimen.

## 4. Inspect a trained model (read-only)

```powershell
$gi = "claude\GenesisInspect\bin\Release\net8.0-windows\GenesisInspect.exe"
& $gi report --cpu                                  # DEFAULT: the daemon model in .claude-nova/
& $gi query "find diagnostic cli" --cpu             # output + decision path + platonic activation
& $gi report --cpu --state-dir "$env:LOCALAPPDATA\GenesisNova\models"   # the autonomous trainer instead
```

GenesisInspect never writes — it loads the same runtime inference uses, so what it shows is what the model does.

## Where to read next

- `README.md` — the engine spec. `CLAUDE.md` — the working agreement (substrate, golden training paths, memory rules).
- `PLATONIC_SPACE.md` — substrate capabilities. `PLATONIC_SHAPES.md` / `PROJECT_GLIDER.md` — the composer.
- `SPACE_AWARE_GRU.md` — the space-aware (perceive→decide→modify→verify) design.
- `claude/README.md` — the applied tools in depth. `claude/SELF_IMPROVE.md` — the compounding self-improvement loop.
