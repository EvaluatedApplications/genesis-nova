# Genesis Nova

Genesis Nova is a compact ML research runtime for learning symbolic and conversational transforms on top of EvalApp.

## What this release contains

- An immutable-record training pipeline
- A token-based neural model with inference-time routing
- Platonic memory / concept tracking for learned structure
- A REPL for training, querying, and inspection
- A UI training panel for model compression presets
- Checkpoint persistence and local conversation memory
- Genesis-focused tests covering training, inference, and discovery

## What changed in this iteration

- Removed route labels from the training data contract
- Removed the old supervised route head path from training
- Kept routing as an inference-time behavior
- Added optional L2 compression, but defaulted it to `0.0`
- Added UI and REPL controls so compression can be enabled later
- Removed the legacy introspection engine and related dead surfaces

## Architecture

The system is organized around a few core pieces:

- `src/GenesisNova/Data/` defines training examples and data generators
- `src/GenesisNova/Train/` loads examples, shapes batches, and runs optimization
- `src/GenesisNova/Model/` owns the neural model, training step, and inference primitives
- `src/GenesisNova/Cognition/` tracks concepts, transforms, and platonic-space structure
- `src/GenesisNova/Infer/` decides how a request is answered at runtime
- `src/GenesisNova/Runtime/` ties together CLI, memory, and checkpoint state
- `src/GenesisNova/UI/` exposes the training controls in the desktop UI

The design goal is to keep the model representation and inference path central, while moving away from hand-coded routing heuristics and dead complexity.

## Training flow

1. Read a `GenesisExample`
2. Tokenize input and target text
3. Run the neural training step
4. Update platonic memory and transform discovery
5. Clone parameters when needed to break the autograd graph

Compression is opt-in only. The default setting is `L2 = 0.0` so learning is not constrained during normal training.

## Inference flow

1. Encode the request
2. Decide whether the request can be answered directly or should use learned structure
3. Generate tokens
4. Optionally consult platonic memory and checkpoint context

## Controls

### REPL

Run:

```bash
dotnet run --project src\GenesisNova.csproj -- --genesis-repl
```

Useful commands include:

- `train <input> => <output>`
- `trainfile <path> [epochs]`
- `config l2`
- `config l2 <preset>`
- `context`
- `compact`
- `reset`

### UI

The Training tab includes L2 preset selection:

- `off`
- `mild`
- `balanced`
- `aggressive`
- `extreme`

## Build and test

```bash
dotnet build GenesisNova.slnx
dotnet test GenesisNova.slnx
```

## Data

Training examples live in `data/`. The main corpus is:

```text
data\genesis-nova-train-expanded.txt
```

