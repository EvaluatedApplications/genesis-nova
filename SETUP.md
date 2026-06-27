# Genesis-Nova — setup guide

**Genesis-Nova is the source of truth for everything in this repo.** The engine in `src/` (project
`GenesisNova.csproj`) is the *general* substrate (a small GRU controller over a structured platonic space — see
`README.md` for the spec and `CLAUDE.md` for the working agreement). It is a `net8.0-windows` WinForms desktop app
that hosts the live model + training gym. Everything else builds on it and must not fork it:

- `src/` (`GenesisNova.csproj`) — **the engine** (GENERAL; no app-specific hardcoding). The single source of truth.
- `Tests/` — the behaviour + emergence suite (in `GenesisNova.slnx`).
- `bench/RaceBench/` — equal-param benchmark vs a transformer (references the engine; also in the .slnx).
- `claude/` — **applied** tooling on top of the engine (NOT in the .slnx, so it never affects the test suite):
  - `GenesisInspect/` — read-only diagnostic CLI: what a trained model *is* and how it answers (the only CLI).
- `.claude-nova/` — **runtime state only** (a model checkpoint + logs, ~400 MB). Generated, gitignored,
  regenerable. Treat as a cache, never as source.

## Prerequisites

- **.NET 8 SDK** (Windows; the projects target `net8.0-windows`).
- **TorchSharp** pulls libtorch automatically on restore. **CPU works everywhere**; an NVIDIA **GPU + CUDA** is
  optional and used for training (`--gpu`) and benchmarks. libtorch prints benign `.grad` warnings to stderr
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

## 2. Build the applied tool

It references the engine but is **not** in `GenesisNova.slnx`, so build it directly once, then run the exe:

```powershell
dotnet build claude/GenesisInspect -c Release
```

## 3. Memory & continuous training

`claude/GenesisInspect` is the **only** CLI (read-only diagnostics — see §4). There is no memory daemon: the
**file memory (`MEMORY.md`) is the source of truth** and is read directly. Continuous skill training now lives
**inside the GenesisNova desktop app's gym** (it runs on startup and trains the live model), not a background
CLI process. See `[[nova-gym]]` for the gym.

## 4. Inspect a trained model (read-only)

```powershell
$gi = "claude\GenesisInspect\bin\Release\net8.0-windows\GenesisInspect.exe"
& $gi report --cpu                                  # DEFAULT: the model in .claude-nova/
& $gi query "find diagnostic cli" --cpu             # output + decision path + platonic activation
& $gi report --cpu --state-dir "$env:LOCALAPPDATA\GenesisNova\gym"   # the live desktop-app gym checkpoint instead
```

GenesisInspect never writes — it loads the same runtime inference uses, so what it shows is what the model does.

## Where to read next

- `README.md` — the engine spec. `CLAUDE.md` — the working agreement (substrate, golden training paths, memory rules).
- `PLATONIC_NUCLEUS.md` — substrate capabilities (the dual-face data model).
- `claude/README.md` — the applied tools in depth. `claude/SELF_IMPROVE.md` — the compounding self-improvement loop.
