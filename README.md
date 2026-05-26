# Genesis Nova

Genesis Nova is an axioms-first ML research runtime implemented on EvalApp.  
This repository now contains only the Genesis Nova model/training/inference/introspection stack and related tests.

## Project layout

- `src/GenesisNova/` - model, trainer, cognition, runtime, REPL, persistence
- `data/` - training corpora
- `models/` - checkpoints and logs
- `docs/GENESIS-NOVA-AXIOMS-BRIDGE.md` - axioms-to-architecture mapping
- `Tests/GenesisNova/` - Genesis-focused test suite

## Build and test

```bash
dotnet build GenesisNova.slnx
dotnet test GenesisNova.slnx
```

## Train

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-train --file data\genesis-nova-train-expanded.txt --epochs 9 --introspect-cycles 64 --threads 16 --log models\genesis-nova.train.log --eval-samples 800
```

The runtime now auto-saves to local app data and resumes the latest checkpoint by default.
Use `--threads <n>` to raise CPU parallelism, or `--no-parallel-math` to disable parallel matrix math.
The REPL keeps a persistent conversation memory and exposes `context`, `compact`, and `reset`.

For rollout safety:

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-train --file data\genesis-nova-train-expanded.txt --epochs 32 --deterministic --backend cpu --baseline-checkpoint models\genesis-nova-v23-e240-i1-t16.checkpoint.json --max-exact-drop 0.01 --eval-samples 800
```

- `--deterministic` forces single-thread deterministic execution.
- `--baseline-checkpoint` compares new model accuracy to a known baseline.
- `--max-exact-drop` fails the run if exact-match drops more than allowed ratio.
- `--backend gpu` auto-scales hidden size against available VRAM unless you pass `--no-auto-scale-vram`.

## REPL

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-repl
```

REPL commands:

```text
introspect [cycles]
concept <name>
relate <left> <right> <contradiction0to1>
queue
```
